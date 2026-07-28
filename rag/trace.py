"""
Диагностика ранжирования: путь целевого чанка через все стадии поиска.

ТОЛЬКО ИЗМЕРЕНИЕ. Модуль ничего не чинит и не меняет поведение обычного пути:
SQL-константы и функция слияния импортируются из rag.retriever как есть,
поэтому цифры здесь — те же, что видит боевой запрос. Генерация в OpenAI
не вызывается: нас интересует только поиск.

Использование:
  python3 -m rag.trace --ids gs001,gs002,gs003,gs006,gs008,gs011
  python3 -m rag.trace --sparse-tokens "Какой номер детали у Hydrosolv 4165?"
  python3 -m rag.trace --scores-csv rerank_scores.csv --ids gs001,gs002,...

Целевой чанк задаётся ПРЕДИКАТОМ ПО СОДЕРЖИМОМУ, а не id: id меняются при
переиндексации, а «чанк, где в content есть 1U-5490» — нет.
"""
from __future__ import annotations

import asyncio
import csv
import math
import sys
import time
from pathlib import Path

import click
import psycopg
import yaml

from .config import settings
from .embedder import Embedder, MAX_LEN_QUERY, sparse_to_pgvector_literal
from .retriever import _DENSE_SQL, _SPARSE_SQL, _FETCH_SQL, _rrf, RetrievedChunk

# Предикат целевого чанка для каждого вопроса golden set.
# Формулировки предикатов — из постановки задачи: для Hydrosolv «где в content
# есть 1U-5490», для air shutoff «heading начинается на Air Shutoff - Test
# и control_module = ...».
TARGETS: dict[str, tuple[str, str]] = {
    "gs001": ("content LIKE '%1U-5490%'", "чанк с таблицей Hydrosolv (Aftercooler Core)"),
    "gs002": ("content LIKE '%1U-5490%'", "чанк с таблицей Hydrosolv (Aftercooler Core)"),
    "gs003": ("content LIKE '%1U-5490%'", "чанк с таблицей Hydrosolv (Aftercooler Core)"),
    "gs006": ("content LIKE '%1U-5490%'", "чанк с таблицей Hydrosolv (Aftercooler Core)"),
    "gs008": ("heading LIKE 'Air Shutoff - Test%' AND control_module = 'ADEM II'",
              "Air Shutoff - Test, ADEM II"),
    "gs011": ("heading LIKE 'Air Shutoff - Test%' AND control_module = 'ADEM III'",
              "Air Shutoff - Test, ADEM III"),
}

_TARGET_SQL = """
SELECT id, heading, page_start, page_end, chunk_type, control_module
FROM chunks
WHERE {predicate}
ORDER BY id
"""


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _load_golden(path: str = "eval/golden_set.yaml") -> dict[str, dict]:
    items = yaml.safe_load(Path(path).read_text())
    return {i["id"]: i for i in items}


async def _stages(query: str) -> dict:
    """
    Прогнать запрос через dense / sparse / RRF и вернуть сырые списки.

    Возвращает полные ранжированные списки со скорами, а НЕ обрезанные до
    rrf_top_k: чтобы можно было сказать «целевой чанк на 34-м месте RRF»,
    а не просто «не попал в топ-20».
    """
    embedder = Embedder()
    t0 = time.monotonic()
    enc = embedder.encode_one(query, max_length=MAX_LEN_QUERY)
    encode_s = time.monotonic() - t0

    dense_list = enc.dense.tolist()
    sparse_literal = sparse_to_pgvector_literal(enc.sparse)

    async with await psycopg.AsyncConnection.connect(settings.db_dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                _DENSE_SQL,
                (dense_list, None, None, None, None, dense_list, settings.dense_top_k),
            )
            dense = [(r[0], float(r[1])) for r in await cur.fetchall()]

            await cur.execute(
                _SPARSE_SQL,
                (sparse_literal, None, None, None, None,
                 sparse_literal, settings.sparse_top_k),
            )
            sparse = [(r[0], float(r[1])) for r in await cur.fetchall()]

            fused = _rrf([[i for i, _ in dense], [i for i, _ in sparse]], k=settings.rrf_k)
            top_ids = [cid for cid, _ in fused[: settings.rrf_top_k]]

            chunks: list[RetrievedChunk] = []
            if top_ids:
                await cur.execute(_FETCH_SQL, (top_ids,))
                by_id = {}
                for row in await cur.fetchall():
                    (id_, heading, icode, section, page_start, page_end, chunk_type,
                     content, smcs, pns, models, cm, step_count, part_index,
                     part_total, safety, has_warning) = row
                    by_id[id_] = RetrievedChunk(
                        id=id_, heading=heading, icode=icode, section=section,
                        page_start=page_start, page_end=page_end, chunk_type=chunk_type,
                        content=content, smcs_codes=smcs or [], part_numbers=pns or [],
                        applicable_models=models or [], control_module=cm,
                        step_count=step_count, part_index=part_index,
                        part_total=part_total, safety_blocks=safety or [],
                        has_warning=has_warning,
                    )
                chunks = [by_id[i] for i in top_ids if i in by_id]

    return {
        "encode_s": encode_s,
        "dense": dense,
        "sparse": sparse,
        "fused": fused,
        "chunks": chunks,
        "enc": enc,
    }


def _rerank_scores(query: str, chunks: list[RetrievedChunk]) -> tuple[list[float], float]:
    """
    Сырые логиты реранкера для кандидатов, в порядке chunks.

    Без сигмоиды: она монотонна, ранга не меняет, а разделение «есть ответ /
    нет ответа» мерялось именно в логитах.

    Через Reranker.logits, а не через _model напрямую: при RERANKER_BACKEND=onnx
    поля _model просто нет, и прежний прямой вызов ронял трассировку.
    """
    from .reranker import Reranker
    return Reranker().logits(query, chunks)


def _rank_of(ranked: list[tuple[int, float]], target_ids: set[int]) -> dict[int, tuple[int, float]]:
    """{target_id: (ранг с 1, скор)} для тех целей, что нашлись в списке."""
    out = {}
    for pos, (cid, score) in enumerate(ranked):
        if cid in target_ids:
            out[cid] = (pos + 1, score)
    return out


async def _trace_one(qid: str, item: dict, do_rerank: bool = True) -> None:
    query = item["question"]
    predicate, human = TARGETS[qid]

    async with await psycopg.AsyncConnection.connect(settings.db_dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(_TARGET_SQL.format(predicate=predicate))
            targets = await cur.fetchall()

    target_ids = {t[0] for t in targets}

    click.secho(f"\n{'='*78}", fg="cyan")
    click.secho(f"{qid}  «{query}»", fg="cyan", bold=True)
    click.secho(f"{'='*78}", fg="cyan")
    click.echo(f"ожидается в ответе: {item.get('expected_contains')}")
    click.echo(f"целевой чанк ({human}), предикат: {predicate}")
    for tid, heading, ps, pe, ctype, cm in targets:
        click.echo(f"   → id={tid}  «{heading[:60]}»  стр. {ps}–{pe}  {ctype}  cm={cm or '—'}")

    st = await _stages(query)

    # ── Dense ────────────────────────────────────────────────────────────────
    dense_hits = _rank_of(st["dense"], target_ids)
    top1_id, top1_score = st["dense"][0] if st["dense"] else (None, 0.0)
    click.secho(f"\nDense top-{settings.dense_top_k}:", bold=True)
    click.echo(f"  топ-1: id={top1_id} score={top1_score:.4f}")
    for tid in sorted(target_ids):
        if tid in dense_hits:
            rank, score = dense_hits[tid]
            click.echo(f"  цель id={tid}: ранг {rank}/{len(st['dense'])}  score={score:.4f}"
                       f"  (отставание от топ-1: {top1_score - score:+.4f})")
        else:
            click.secho(f"  цель id={tid}: НЕ НАЙДЕН в top-{settings.dense_top_k}", fg="red")

    # ── Sparse ───────────────────────────────────────────────────────────────
    sparse_hits = _rank_of(st["sparse"], target_ids)
    s_top1_id, s_top1_score = st["sparse"][0] if st["sparse"] else (None, 0.0)
    click.secho(f"\nSparse top-{settings.sparse_top_k}:", bold=True)
    click.echo(f"  топ-1: id={s_top1_id} score={s_top1_score:.4f}")
    for tid in sorted(target_ids):
        if tid in sparse_hits:
            rank, score = sparse_hits[tid]
            click.echo(f"  цель id={tid}: ранг {rank}/{len(st['sparse'])}  score={score:.4f}"
                       f"  (отставание от топ-1: {s_top1_score - score:+.4f})")
        else:
            click.secho(f"  цель id={tid}: НЕ НАЙДЕН в top-{settings.sparse_top_k}", fg="red")

    # ── RRF ──────────────────────────────────────────────────────────────────
    rrf_hits = _rank_of(st["fused"], target_ids)
    click.secho(f"\nRRF (k={settings.rrf_k}) → top-{settings.rrf_top_k}:", bold=True)
    click.echo(f"  всего кандидатов после слияния: {len(st['fused'])}")
    for tid in sorted(target_ids):
        d_rank = dense_hits.get(tid, (None, None))[0]
        s_rank = sparse_hits.get(tid, (None, None))[0]
        d_contrib = 1.0 / (settings.rrf_k + d_rank) if d_rank else 0.0
        s_contrib = 1.0 / (settings.rrf_k + s_rank) if s_rank else 0.0
        if tid in rrf_hits:
            rank, score = rrf_hits[tid]
            inside = rank <= settings.rrf_top_k
            click.secho(
                f"  цель id={tid}: ранг {rank}/{len(st['fused'])}  rrf={score:.5f}"
                f"  {'(в top-20)' if inside else '(ВНЕ top-20 — отсечён)'}",
                fg="green" if inside else "red",
            )
            click.echo(f"      вклад dense:  {d_contrib:.5f}  (ранг {d_rank or '—'})")
            click.echo(f"      вклад sparse: {s_contrib:.5f}  (ранг {s_rank or '—'})")
        else:
            click.secho(f"  цель id={tid}: НЕ НАЙДЕН ни в одной ветви", fg="red")

    if not do_rerank:
        return

    # ── Реранкер ─────────────────────────────────────────────────────────────
    chunks = st["chunks"]
    click.secho(f"\nРеранкер: {len(chunks)} кандидатов → top-{settings.rerank_top_n}:", bold=True)
    if not chunks:
        click.secho("  кандидатов нет", fg="red")
        return
    scores, elapsed = _rerank_scores(query, chunks)
    ranked = sorted(zip(scores, chunks), key=lambda x: -x[0])
    click.echo(f"  ({elapsed:.1f} с)")
    top_logit = ranked[0][0]
    click.echo(f"  топ-1: logit={top_logit:+.4f}  sigmoid={_sigmoid(top_logit):.4f}"
               f"  id={ranked[0][1].id} «{ranked[0][1].heading[:50]}»")
    for tid in sorted(target_ids):
        pos = next((i for i, (_, c) in enumerate(ranked) if c.id == tid), None)
        if pos is None:
            click.secho(f"  цель id={tid}: не дошла до реранкера (отсечена на RRF)", fg="red")
        else:
            lg = ranked[pos][0]
            inside = pos < settings.rerank_top_n
            click.secho(
                f"  цель id={tid}: ранг {pos+1}/{len(ranked)}  logit={lg:+.4f}"
                f"  sigmoid={_sigmoid(lg):.4f}  {'(в top-6)' if inside else '(ВНЕ top-6)'}",
                fg="green" if inside else "red",
            )

    click.secho(f"\nИтоговая шестёрка в контекст генерации:", bold=True)
    for i, (sc, c) in enumerate(ranked[: settings.rerank_top_n]):
        mark = " ←ЦЕЛЬ" if c.id in target_ids else ""
        click.echo(
            f"  [{i+1}] id={c.id:<4} logit={sc:+.4f} sig={_sigmoid(sc):.4f}  "
            f"стр. {c.page_start}–{c.page_end}  {c.chunk_type:<9} «{c.heading[:48]}»{mark}"
        )


def _sparse_tokens(query: str, top_n: int = 15) -> None:
    """Топ-N токенов разрежённого представления запроса с расшифровкой."""
    embedder = Embedder()
    enc = embedder.encode_one(query, max_length=MAX_LEN_QUERY)
    embedder._load()
    tok = embedder._model.tokenizer

    # sparse: {pgvec_index (1-based): weight} → token_id = idx - 1
    items = sorted(enc.sparse.items(), key=lambda kv: -kv[1])[:top_n]
    click.secho(f"\nSparse-представление запроса «{query}»", fg="cyan", bold=True)
    click.echo(f"ненулевых токенов всего: {len(enc.sparse)}")
    click.echo(f"\n{'#':>3}  {'token_id':>8}  {'вес':>8}  токен")
    for i, (idx, w) in enumerate(items):
        token_id = idx - 1
        token = tok.convert_ids_to_tokens(token_id)
        click.echo(f"{i+1:>3}  {token_id:>8}  {w:>8.4f}  {token!r}")


async def _scores_csv(ids: list[str], golden: dict[str, dict], out_path: str) -> None:
    """
    Скоры реранкера по вопросам — для калибровки порога честного отказа.

    Генерация не вызывается. Пишем топ-1 и топ-6 логиты и сигмоиды.
    """
    rows = []
    for n, qid in enumerate(ids, 1):
        item = golden[qid]
        query = item["question"]
        click.echo(f"  [{n}/{len(ids)}] {qid}: {query[:50]}…", err=True)
        st = await _stages(query)
        chunks = st["chunks"]
        if not chunks:
            continue
        scores, elapsed = _rerank_scores(query, chunks)
        ranked = sorted(scores, reverse=True)
        top6 = ranked[: settings.rerank_top_n]
        rows.append({
            "id": qid,
            "category": item.get("category", ""),
            "question": query,
            "n_candidates": len(chunks),
            "top1_logit": f"{ranked[0]:.4f}",
            "top1_sigmoid": f"{_sigmoid(ranked[0]):.4f}",
            "top6_logit": f"{top6[-1]:.4f}",
            "top6_sigmoid": f"{_sigmoid(top6[-1]):.4f}",
            "mean_top6_logit": f"{sum(top6)/len(top6):.4f}",
            "rerank_s": f"{elapsed:.1f}",
        })
        click.echo(f"      top1={ranked[0]:+.3f}  top6={top6[-1]:+.3f}  ({elapsed:.0f} с)", err=True)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    click.secho(f"\nЗаписано: {out_path} ({len(rows)} строк)", fg="green")


@click.command()
@click.option("--ids", default=None, help="ID через запятую: gs001,gs002")
@click.option("--sparse-tokens", "sparse_q", default=None, help="Показать sparse-токены запроса")
@click.option("--scores-csv", "csv_path", default=None, help="Собрать скоры реранкера в CSV")
@click.option("--no-rerank", is_flag=True, help="Только dense/sparse/RRF, без реранкера")
@click.option("--golden-set", default="eval/golden_set.yaml", show_default=True)
def main(ids, sparse_q, csv_path, no_rerank, golden_set):
    if sparse_q:
        _sparse_tokens(sparse_q)
        return

    golden = _load_golden(golden_set)
    id_list = [s.strip() for s in ids.split(",")] if ids else []

    if csv_path:
        if not id_list:
            id_list = list(golden.keys())
        asyncio.run(_scores_csv(id_list, golden, csv_path))
        return

    if not id_list:
        click.echo("Укажи --ids или --sparse-tokens.", err=True)
        raise SystemExit(1)

    for qid in id_list:
        if qid not in TARGETS:
            click.secho(f"{qid}: предикат целевого чанка не задан в TARGETS — пропускаю", fg="yellow")
            continue
        asyncio.run(_trace_one(qid, golden[qid], do_rerank=not no_rerank))


if __name__ == "__main__":
    main()
