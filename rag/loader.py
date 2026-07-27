"""
Загрузка chunks.json → PostgreSQL.

Алгоритм:
  1. Для каждого чанка генерируем contextual prefix через gpt-4o-mini
     (1-2 предложения: «фрагмент из O&M Manual SEBU7844-37, секция ..., процедура ...»).
  2. Кэшируем префиксы на диск по SHA-256 содержимого — переиндексация не платит повторно.
  3. Эмбеддируем: context_prefix + "\n\n" + content.
  4. Перед вставкой удаляем старые чанки документа (дедупликация через DELETE).
  5. Вставляем батчами, коммитим каждые 20 чанков.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path

import psycopg
from openai import AsyncOpenAI

from .config import settings
from .embedder import Embedder, MAX_LEN_INDEX, sparse_to_pgvector_literal

log = logging.getLogger(__name__)

_embedder = Embedder()

_PREFIX_SYSTEM = (
    "You are a technical documentation assistant. Generate 1–2 sentences "
    "describing what context this fragment comes from. Be specific about "
    "the document title, section, and procedure topic. English only. "
    "No preamble — start directly with the description."
)


# ─── Prefix cache ─────────────────────────────────────────────────────────────

def _load_cache(path: str) -> dict[str, str]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def _save_cache(cache: dict[str, str], path: str) -> None:
    Path(path).write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def _chunk_key(chunk: dict) -> str:
    return hashlib.sha256(chunk["content"].encode()).hexdigest()[:16]


async def _generate_prefix(chunk: dict, client: AsyncOpenAI) -> str:
    pages = (
        f"p. {chunk['page_start']}"
        if chunk["page_start"] == chunk["page_end"]
        else f"pp. {chunk['page_start']}–{chunk['page_end']}"
    )
    user_msg = (
        f"Document: SEBU7844-37 — Caterpillar 3500B/3500C Marine Engine "
        f"Operation & Maintenance Manual\n"
        f"Section: {chunk.get('section') or 'General'}\n"
        f"Heading: {chunk['heading']}\n"
        f"Type: {chunk['chunk_type']}, pages: {pages}\n\n"
        f"Text excerpt:\n{chunk['content'][:600]}"
    )
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _PREFIX_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=80,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


async def _get_or_generate_prefix(
    chunk: dict,
    cache: dict[str, str],
    client: AsyncOpenAI,
) -> str:
    key = _chunk_key(chunk)
    if key in cache:
        return cache[key]
    prefix = await _generate_prefix(chunk, client)
    cache[key] = prefix
    return prefix


# ─── DB helpers ───────────────────────────────────────────────────────────────

async def _ensure_document(conn: psycopg.AsyncConnection, filename: str) -> int:
    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO documents (filename)
            VALUES (%s)
            ON CONFLICT (filename) DO UPDATE SET indexed_at = now()
            RETURNING id
            """,
            (filename,),
        )
        row = await cur.fetchone()
        return row[0]


async def _delete_chunks(conn: psycopg.AsyncConnection, doc_id: int) -> int:
    async with conn.cursor() as cur:
        await cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        return cur.rowcount


async def _insert_chunk(
    conn: psycopg.AsyncConnection,
    doc_id: int,
    chunk: dict,
    context_prefix: str,
    dense: "np.ndarray",
    sparse: dict[int, float],
) -> None:
    sparse_literal = sparse_to_pgvector_literal(sparse)

    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO chunks (
                doc_id, heading, icode, section,
                page_start, page_end, chunk_type, content,
                smcs_codes, part_numbers, illustrations, applicable_models,
                control_module, step_count, part_index, part_total,
                safety_blocks, has_warning,
                context_prefix, embedding_dense, embedding_sparse
            ) VALUES (
                %(doc_id)s, %(heading)s, %(icode)s, %(section)s,
                %(page_start)s, %(page_end)s, %(chunk_type)s, %(content)s,
                %(smcs_codes)s, %(part_numbers)s, %(illustrations)s, %(applicable_models)s,
                %(control_module)s, %(step_count)s, %(part_index)s, %(part_total)s,
                %(safety_blocks)s::jsonb, %(has_warning)s,
                %(context_prefix)s,
                %(embedding_dense)s::vector,
                %(embedding_sparse)s::sparsevec
            )
            """,
            {
                "doc_id": doc_id,
                "heading": chunk["heading"],
                "icode": chunk.get("icode"),
                "section": chunk.get("section"),
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "chunk_type": chunk["chunk_type"],
                "content": chunk["content"],
                "smcs_codes": chunk.get("smcs_codes", []),
                "part_numbers": chunk.get("part_numbers", []),
                "illustrations": chunk.get("illustrations", []),
                "applicable_models": chunk.get("applicable_models", []),
                "control_module": chunk.get("control_module"),
                "step_count": chunk.get("step_count", 0),
                "part_index": chunk.get("part_index", 0),
                "part_total": chunk.get("part_total", 1),
                "safety_blocks": json.dumps(chunk.get("safety_blocks", [])),
                "has_warning": chunk.get("has_warning", False),
                "context_prefix": context_prefix,
                "embedding_dense": dense.tolist(),
                "embedding_sparse": sparse_literal,
            },
        )


# ─── Основная точка входа ─────────────────────────────────────────────────────

async def load(
    chunks_path: str | None = None,
    filename: str = "SEBU7844-37.pdf",
) -> None:
    chunks_path = chunks_path or settings.chunks_path
    chunks: list[dict] = json.loads(Path(chunks_path).read_text())
    cache = _load_cache(settings.prefix_cache_path)
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # 1. Contextual prefixes (с кэшированием)
    log.info("Генерирую контекстуальные префиксы (%d чанков)…", len(chunks))
    for i, chunk in enumerate(chunks):
        chunk["_prefix"] = await _get_or_generate_prefix(chunk, cache, client)
        if (i + 1) % 20 == 0:
            _save_cache(cache, settings.prefix_cache_path)
            log.info("  %d/%d префиксов готово", i + 1, len(chunks))
    _save_cache(cache, settings.prefix_cache_path)
    log.info("Все префиксы готовы (кэш: %s)", settings.prefix_cache_path)

    # 2. Эмбеддинги: prefix + "\n\n" + content
    embed_texts = [f"{c['_prefix']}\n\n{c['content']}" for c in chunks]
    log.info("Эмбеддирую %d текстов (batch=%d, max_length=%d)…",
             len(chunks), settings.embed_batch_size, MAX_LEN_INDEX)
    embed_results = _embedder.encode(
        embed_texts,
        batch_size=settings.embed_batch_size,
        max_length=MAX_LEN_INDEX,
    )
    log.info("Эмбеддинги готовы.")

    # 3. Вставка в БД
    async with await psycopg.AsyncConnection.connect(settings.db_dsn) as conn:
        doc_id = await _ensure_document(conn, filename)
        deleted = await _delete_chunks(conn, doc_id)
        if deleted:
            log.info("Удалено %d старых чанков для doc_id=%d", deleted, doc_id)
        await conn.commit()

        log.info("Вставляю %d чанков (doc_id=%d)…", len(chunks), doc_id)
        for i, (chunk, result) in enumerate(zip(chunks, embed_results)):
            await _insert_chunk(
                conn, doc_id, chunk, chunk["_prefix"],
                result.dense, result.sparse,
            )
            if (i + 1) % 20 == 0:
                await conn.commit()
                log.info("  %d/%d чанков вставлено", i + 1, len(chunks))

        await conn.commit()
        log.info("Загрузка завершена: %d чанков в PostgreSQL.", len(chunks))
