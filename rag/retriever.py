"""
Гибридный поиск: dense ANN + sparse sequential scan → RRF → top-20.

Фильтры по метаданным (applicable_models, control_module) применяются в SQL WHERE,
а не в промпте — это правило 7 из CLAUDE.md.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import psycopg

from .config import settings
from .embedder import Embedder, MAX_LEN_QUERY, sparse_to_pgvector_literal

log = logging.getLogger(__name__)
_embedder = Embedder()

# %s-параметры: dense_vec, models_arr, models_arr, control_module, control_module, dense_vec, limit
_DENSE_SQL = """
SELECT id, 1 - (embedding_dense <=> %s::vector) AS score
FROM chunks
WHERE
    (%s::text[] IS NULL OR applicable_models && %s::text[])
    AND (%s::text IS NULL OR control_module = %s)
ORDER BY embedding_dense <=> %s::vector
LIMIT %s
"""

# %s-параметры: sparse_literal, models_arr, models_arr, control_module, control_module, sparse_literal, limit
# <#> возвращает −dot_product → ORDER BY <#> = убывание сходства
_SPARSE_SQL = """
SELECT id, -(embedding_sparse <#> %s::sparsevec) AS score
FROM chunks
WHERE
    (%s::text[] IS NULL OR applicable_models && %s::text[])
    AND (%s::text IS NULL OR control_module = %s)
ORDER BY embedding_sparse <#> %s::sparsevec
LIMIT %s
"""

# JOIN к documents добавлен ради applicability документа: строка «Применимо к:»
# в ответе собирается механически, и когда чанк применимость не сузил (а таких
# большинство), берётся применимость документа с титульной страницы.
# На ранжирование не влияет — это дозагрузка полей уже отобранных id,
# порядок и отбор задаются dense/sparse-запросами выше.
_FETCH_SQL = """
SELECT
    c.id, c.heading, c.icode, c.section,
    c.page_start, c.page_end, c.chunk_type, c.content,
    c.smcs_codes, c.part_numbers, c.applicable_models,
    c.control_module, c.step_count, c.part_index, c.part_total,
    c.safety_blocks, c.has_warning,
    d.applicable_models AS doc_models
FROM chunks c
JOIN documents d ON d.id = c.doc_id
WHERE c.id = ANY(%s::int[])
"""


@dataclass
class RetrievedChunk:
    id: int
    heading: str
    icode: str | None
    section: str | None
    page_start: int
    page_end: int
    chunk_type: str
    content: str
    smcs_codes: list[str]
    part_numbers: list[str]
    applicable_models: list[str]
    control_module: str | None
    step_count: int
    part_index: int
    part_total: int
    safety_blocks: list[dict]
    has_warning: bool
    # Применимость документа с титульной страницы — запасной уровень для
    # строки «Применимо к:», когда чанк не сузил её сам. Не участвует
    # ни в поиске, ни в ранжировании.
    doc_models: list[str] = field(default_factory=list)
    rrf_score: float = 0.0

    @property
    def citation(self) -> str:
        if self.page_start == self.page_end:
            return f"стр. {self.page_start}"
        return f"стр. {self.page_start}–{self.page_end}"


def _rrf(ranked_lists: list[list[int]], k: int) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion — стандартная k=60."""
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, chunk_id in enumerate(lst):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


async def retrieve(
    query: str,
    models: list[str] | None = None,
    control_module: str | None = None,
) -> list[RetrievedChunk]:
    """
    Гибридный поиск.

    models: список моделей движка для фильтрации, например ['3516B', '3516C'].
            None = без фильтра (ищем по всем документам).
    control_module: 'ADEM II' | 'ADEM III' | None.
    """
    t0 = time.monotonic()
    result = _embedder.encode_one(query, max_length=MAX_LEN_QUERY)
    log.debug("Запрос закодирован за %.2f с", time.monotonic() - t0)

    dense_list = result.dense.tolist()
    sparse_literal = sparse_to_pgvector_literal(result.sparse)

    async with await psycopg.AsyncConnection.connect(settings.db_dsn) as conn:
        async with conn.cursor() as cur:

            # Dense ANN
            await cur.execute(
                _DENSE_SQL,
                (dense_list, models, models,
                 control_module, control_module,
                 dense_list, settings.dense_top_k),
            )
            dense_ids = [r[0] for r in await cur.fetchall()]

            # Sparse (sequential scan — достаточно при <10k чанков)
            await cur.execute(
                _SPARSE_SQL,
                (sparse_literal, models, models,
                 control_module, control_module,
                 sparse_literal, settings.sparse_top_k),
            )
            sparse_ids = [r[0] for r in await cur.fetchall()]

            log.debug(
                "dense=%d sparse=%d candidates",
                len(dense_ids), len(sparse_ids),
            )

            # RRF
            fused = _rrf([dense_ids, sparse_ids], k=settings.rrf_k)
            top = fused[: settings.rrf_top_k]
            if not top:
                return []
            top_ids = [chunk_id for chunk_id, _ in top]
            id_to_score = {chunk_id: score for chunk_id, score in top}

            # Забираем данные чанков одним запросом
            await cur.execute(_FETCH_SQL, (top_ids,))
            rows = await cur.fetchall()

    chunk_map: dict[int, RetrievedChunk] = {}
    for row in rows:
        (
            id_, heading, icode, section,
            page_start, page_end, chunk_type, content,
            smcs_codes, part_numbers, applicable_models,
            control_module_, step_count, part_index, part_total,
            safety_blocks, has_warning, doc_models,
        ) = row
        chunk_map[id_] = RetrievedChunk(
            id=id_,
            heading=heading,
            icode=icode,
            section=section,
            page_start=page_start,
            page_end=page_end,
            chunk_type=chunk_type,
            content=content,
            smcs_codes=smcs_codes or [],
            part_numbers=part_numbers or [],
            applicable_models=applicable_models or [],
            control_module=control_module_,
            step_count=step_count,
            part_index=part_index,
            part_total=part_total,
            safety_blocks=safety_blocks or [],
            has_warning=has_warning,
            doc_models=doc_models or [],
            rrf_score=id_to_score.get(id_, 0.0),
        )

    log.info("retrieve: %d чанков за %.2f с", len(chunk_map), time.monotonic() - t0)
    return [chunk_map[i] for i in top_ids if i in chunk_map]
