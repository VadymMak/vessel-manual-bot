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

# РАСКРЫТИЕ СЕМЕЙСТВ МОДЕЛЕЙ.
#
# Caterpillar печатает «3500B» как имя СЕМЕЙСТВА, а не модели. Титул
# SEBU7844-37 несёт заголовок «3500B and 3500C Marine Engines», а ниже — строки
# «S2D 1-UP (3508B) … S2J 1-UP (3512B) … S2S 1-UP (3516B)», из которых и собран
# d.applicable_models. Чанк с разметкой {3500B} применим ко всем B-двигателям
# документа, но фильтр сравнивал строки как непрозрачные токены и родства
# не видел: Air Shutoff ADEM II (стр. 68–69) был недостижим ни для одного
# запроса по конкретной модели.
#
# ТАБЛИЦА СЕМЕЙСТВ НЕ ЗАДАНА КОНСТАНТОЙ — она выводится из корпуса:
#   семейный токен = токен в разметке чанка, которого НЕТ в списке моделей
#                    своего документа (3500B среди S2J-моделей нет, 3512B есть);
#   покрытие       = модели того же документа с тем же поколением, где
#                    поколение — хвост после ведущих цифр (3500B → «B»).
# На корпусе из двух мануалов это даёт ровно две строки, обе совпадают
# с титульной страницей и ничего лишнего не приносят:
#   3500B → 3508B, 3512B, 3516B
#   3500C → 3508C, 3512C, 3516C
# C18 семейством не становится: он присутствует в списке моделей своего
# документа. Новый мануал даёт свои строки без правки кода — см.
# scripts/family_table.py.
#
# НАПРАВЛЕНИЕ РАСКРЫТИЯ ОДНО: модель → семейства, её содержащие. Запрос 3512B
# расширяется до {3512B, 3500B}. Обратное направление — запрос по семейству,
# подхватывающий чанки конкретных моделей, — СОЗНАТЕЛЬНО НЕ РЕАЛИЗОВАНО:
# это другая операция с другой ценой ошибки, и решать её надо отдельно.
#
# ПОКОЛЕНИЕ ОБЯЗАНО СОВПАДАТЬ. {3500C} под запросом 3512B не проходит:
# Fluid Recommendations стр. 56–63 размечен {3500C} и B-двигателю не применим.
# Цена ошибки здесь та же, что у ADEM II против ADEM III.
_FAMILY_CTE = """
WITH fam AS (
    SELECT DISTINCT unnest(c.applicable_models) AS token, c.doc_id
    FROM chunks c
), families AS (
    SELECT f.token, f.doc_id
    FROM fam f
    JOIN documents d ON d.id = f.doc_id
    WHERE NOT (f.token = ANY(d.applicable_models))
), asked AS MATERIALIZED (
    SELECT array_agg(DISTINCT x) AS models FROM (
        SELECT unnest(%(models)s::text[]) AS x
        UNION
        SELECT fa.token
        FROM families fa
        JOIN documents d ON d.id = fa.doc_id
        CROSS JOIN LATERAL unnest(d.applicable_models) AS m
        WHERE m = ANY(%(models)s::text[])
          AND regexp_replace(m,  '^[0-9]+', '')
            = regexp_replace(fa.token, '^[0-9]+', '')
    ) s
)
"""

# ФИЛЬТР ПО МОДЕЛИ ДВИГАТЕЛЯ (правило 7 CLAUDE.md: в SQL WHERE, не в промпте).
#
# Применимость берётся с приоритетом: собственная разметка чанка, если она есть,
# иначе унаследованная от документа. Ровно то же разрешение, что у строки
# «Применимо к:» в rag/generator.py — они обязаны совпадать, иначе ответ будет
# заявлять применимость, по которой его же и не нашли бы.
#
# Три случая, и третий забывали дважды:
#   разметка содержит запрошенную модель            → пройти
#   разметка содержит семейство, её включающее      → пройти (через `asked`)
#   разметка ПУСТА, применимость от документа       → пройти, если модель есть
#                                                     у документа
#   разметка содержит чужие модели или поколения    → отсечь
#
# Фильтровать ПО ЧАНКУ ОДНОМУ НЕЛЬЗЯ: собственную разметку несут 12 чанков
# из 394, у остальных массив пуст, и `applicable_models && ARRAY['3512B']`
# отсеял бы 97% базы вместе с правильными ответами. Пустой массив означает
# «применимо ко всему семейству документа», а не «не применимо ни к чему»,
# и ветка ELSE ниже обрабатывает это ЯВНО, а не по совпадению.
#
# И это НЕ фильтр по документу: чанк C18 с собственной разметкой пройдёт
# фильтр 3512B, если 3512B у него указан, а чанк SEBU7844 с разметкой
# {3500C} под 3512B не пройдёт, хотя документ подходит.
_MODEL_FILTER = """
    (%(models)s::text[] IS NULL OR
     (CASE WHEN cardinality(c.applicable_models) > 0
           THEN c.applicable_models
           ELSE d.applicable_models END) && (SELECT models FROM asked))
"""

# Параметры именованные: dense, models, cm, limit
_DENSE_SQL = f"""
{_FAMILY_CTE}
SELECT c.id, 1 - (c.embedding_dense <=> %(dense)s::vector) AS score
FROM chunks c
JOIN documents d ON d.id = c.doc_id
WHERE
{_MODEL_FILTER}
    AND (%(cm)s::text IS NULL OR c.control_module = %(cm)s)
ORDER BY c.embedding_dense <=> %(dense)s::vector
LIMIT %(limit)s
"""

# Параметры именованные: sparse, models, cm, limit
# <#> возвращает −dot_product → ORDER BY <#> = убывание сходства
#
# `> 0` — НЕ ЛИШНЕЕ УСЛОВИЕ, НЕ СНИМАТЬ. Замер 2026-08-03:
#
#   группа вопросов        ненулевых скоров в топ-50 sparse
#   26 якорных             медиана 50 из 50
#   gs022, gs031           1 из 50
#   10 ru_no_anchor        медиана 0 из 50, у шести ровно 0
#
# На кириллическом запросе лексического пересечения с английским корпусом
# нет, и sparse возвращает ровно ноль почти всему окну. Порядок среди нулей
# Postgres выбирает произвольно — это тай-брейк, а не ранжирование, и он
# не совпадает даже между сессиями: два прогона gs035 из разных соединений
# дали окна с пересечением 0 из 50.
#
# Без этого условия слияние принимало 49 случайных чанков за свидетельства
# второй ветви и топило ими цель, найденную dense: gs033 стоял на втором
# месте dense и уходил на 21-е после слияния, за границу RRF_TOP_K = 20,
# то есть терялся целиком.
#
# И это же условие возвращает воспроизводимость: без него make eval-retrieval
# детерминирован только на вопросах с латинским якорем, а на остальных
# показывает один жребий.
_SPARSE_SQL = f"""
{_FAMILY_CTE}
SELECT c.id, -(c.embedding_sparse <#> %(sparse)s::sparsevec) AS score
FROM chunks c
JOIN documents d ON d.id = c.doc_id
WHERE
{_MODEL_FILTER}
    AND (%(cm)s::text IS NULL OR c.control_module = %(cm)s)
    AND -(c.embedding_sparse <#> %(sparse)s::sparsevec) > 0
ORDER BY c.embedding_sparse <#> %(sparse)s::sparsevec
LIMIT %(limit)s
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
    d.applicable_models AS doc_models,
    d.filename AS doc_filename
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
    # Из какого мануала чанк. Нужен для метрики загрязнения: при нескольких
    # документах в базе доля чужих чанков в топ-6 растёт раньше, чем портятся
    # ответы, — это опережающий индикатор.
    doc_filename: str = ""
    rrf_score: float = 0.0

    @property
    def citation(self) -> str:
        if self.page_start == self.page_end:
            return f"стр. {self.page_start}"
        return f"стр. {self.page_start}–{self.page_end}"


def _rrf(ranked_lists: list[list[int]], k: int,
         miss_rank: int | None = None) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion — стандартная k=60, с ВМЕНЕНИЕМ ОТСУТСТВИЯ.

    Отсутствие кандидата в окне ветви даёт не ноль слагаемых, а одно
    штрафное: ранг `miss_rank`, заведомо хуже последнего места в окне.
    Иначе кандидат, попавший в обе ветви, обыгрывает одноветочного
    по построению формулы, а не по релевантности:

        одноветочный   1/(60+r)
        двуветочный    1/(60+r₁) + 1/(60+r₂)

    Точка безразличия 2/(60+r) = 1/61 даёт r = 62 при окне 50 — то есть
    любой двуветочный кандидат В ОКНЕ обходит одноветочную цель, где бы
    та ни стояла в своей ветви. Не иногда, а всегда. На gs033 это доводило
    до потери ответа: цель на втором месте dense уходила на 21-е при
    RRF_TOP_K = 20.

    Вменение убирает именно это: каждый кандидат получает по слагаемому
    от каждой ветви, и наличие во второй перестаёт быть бонусом само
    по себе. Замер 2026-08-03 на 10 вопросах ru_no_anchor, плечо M="3512B":
    медиана ранга цели 21 → 1, MRR 0.097 → 0.719, при нуле ухудшений
    среди 26 якорных вопросов.

    ТАЙ-БРЕЙК. При вменении кандидат с dense-рангом 1 получает РОВНО тот же
    скор, что кандидат со sparse-рангом 1: 1/61 + 1/111 в обе стороны.
    Порядок между ними решает устойчивость sorted() и порядок наполнения
    словаря — dense-список идёт первым, поэтому при равенстве побеждает он.
    Это поведение было и раньше (там равенство тоже возникало, 1/61 против
    1/61), но теперь на него опирается результат, и менять порядок
    ranked_lists местами нельзя.
    """
    if miss_rank is None:
        miss_rank = max(settings.dense_top_k, settings.sparse_top_k) + 1
    positions = [{cid: r + 1 for r, cid in enumerate(lst)} for lst in ranked_lists]
    # Порядок вставки задаёт тай-брейк: сначала вся первая ветвь, затем
    # то, что есть только во второй. Отсюда «при равенстве побеждает dense».
    scores: dict[int, float] = {}
    for pos in positions:
        for cid in pos:
            scores.setdefault(cid, 0.0)
    for cid in scores:
        scores[cid] = sum(1.0 / (k + pos.get(cid, miss_rank)) for pos in positions)
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
                {"dense": dense_list, "models": models,
                 "cm": control_module, "limit": settings.dense_top_k},
            )
            dense_ids = [r[0] for r in await cur.fetchall()]

            # Sparse (sequential scan — достаточно при <10k чанков)
            await cur.execute(
                _SPARSE_SQL,
                {"sparse": sparse_literal, "models": models,
                 "cm": control_module, "limit": settings.sparse_top_k},
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
            safety_blocks, has_warning, doc_models, doc_filename,
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
            doc_filename=doc_filename or "",
            rrf_score=id_to_score.get(id_, 0.0),
        )

    log.info("retrieve: %d чанков за %.2f с", len(chunk_map), time.monotonic() - t0)
    return [chunk_map[i] for i in top_ids if i in chunk_map]
