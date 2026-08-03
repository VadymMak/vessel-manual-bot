#!/usr/bin/env python3
"""
Выгрузка полных списков рангов dense и sparse для офлайн-перебора слияния.

ЗАЧЕМ. Слияние — чистая функция от двух списков рангов. Перебирать её
варианты прогонами через базу, эмбеддер и реранкер бессмысленно: один прогон
38 вопросов стоит двадцать минут, а перебор двадцати вариантов — семь часов
на вычисление, которое считается за секунду. Выгружаем ранги один раз,
дальше работаем в памяти.

После этого скрипта обращений к базе, эмбеддеру и реранкеру в задаче
про слияние не требуется вовсе.

ПЛЕЧИ:
  nofilter       models=None
  m3512B         models=['3512B'], фильтр с раскрытием семейств — текущий код
  m3512B_noexp   models=['3512B'], фильтр БЕЗ раскрытия — прежнее поведение,
                 нужно ровно для проверки гипотезы по gs031 (что именно
                 изменилось в окнах, когда добавились семь чанков)

Для вопросов ru_no_anchor дополнительно выгружаются ранги по РУКОПИСНЫМ
английским формулировкам (scripts/ru_en_probe.py) — замер пользы перевода.

  python3 -m scripts.dump_ranks               → eval_ranks.json
  python3 -m scripts.dump_ranks --out путь
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import psycopg
import yaml

# Прежний фильтр, ДО раскрытия семейств. Нужен только для сравнения окон
# на gs031 и больше нигде: это не откат, а измерительная копия.
_OLD_FILTER = """
    (%(models)s::text[] IS NULL OR
     (CASE WHEN cardinality(c.applicable_models) > 0
           THEN c.applicable_models
           ELSE d.applicable_models END) && %(models)s::text[])
"""
_OLD_DENSE = f"""
SELECT c.id, 1 - (c.embedding_dense <=> %(dense)s::vector)
FROM chunks c JOIN documents d ON d.id = c.doc_id
WHERE {_OLD_FILTER}
ORDER BY c.embedding_dense <=> %(dense)s::vector LIMIT %(limit)s
"""
_OLD_SPARSE = f"""
SELECT c.id, -(c.embedding_sparse <#> %(sparse)s::sparsevec)
FROM chunks c JOIN documents d ON d.id = c.doc_id
WHERE {_OLD_FILTER}
ORDER BY c.embedding_sparse <#> %(sparse)s::sparsevec LIMIT %(limit)s
"""

_TARGET_SQL = """
SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id
WHERE (%(doc)s::text IS NULL OR d.filename = %(doc)s)
  AND (%(h)s::text IS NULL OR c.heading LIKE %(hl)s)
  AND (%(c)s::text IS NULL OR c.content LIKE %(cl)s)
  AND (%(cm)s::text IS NULL OR c.control_module = %(cm)s)
ORDER BY c.id
"""

# Проходит ли САМА ЦЕЛЬ под фильтром. Нужно, чтобы отличать «поиск
# промахнулся» от «фильтр корректно отсёк чужой мануал»: цели gs040–gs042
# лежат в C18, и под M="3512B" их отсутствие — правильная работа, а не провал.
#
# Раскрытие семейств здесь обязано учитываться ровно так же, как в плече:
# цель gs008 размечена {3500B} и под НЕраскрытым фильтром 3512B не проходит,
# а под раскрытым проходит. Проверка без раскрытия объявила бы gs008
# отсечённым и молча выкинула бы его из группы якорных.
_DOC_PASSES_OLD = """
SELECT bool_or(
    (CASE WHEN cardinality(c.applicable_models) > 0
          THEN c.applicable_models ELSE d.applicable_models END) && %(models)s::text[])
FROM chunks c JOIN documents d ON d.id = c.doc_id
WHERE c.id = ANY(%(ids)s::int[])
"""


async def _targets(cur, spec: dict) -> list[int]:
    h, cc = spec.get("heading_startswith"), spec.get("content_contains")
    await cur.execute(_TARGET_SQL, {
        "doc": spec.get("doc"), "h": h, "hl": f"{h}%" if h else None,
        "c": cc, "cl": f"%{cc}%" if cc else None,
        "cm": spec.get("control_module")})
    return [r[0] for r in await cur.fetchall()]


async def _reachable(cur, models, targets: list[int], expand: bool) -> bool:
    """Проходит ли сама цель под фильтром этого плеча, с тем же раскрытием."""
    from rag.retriever import _FAMILY_CTE, _MODEL_FILTER

    if models is None:
        return True
    if expand:
        sql = _FAMILY_CTE + f"""
        SELECT count(*) > 0 FROM chunks c JOIN documents d ON d.id = c.doc_id
        WHERE {_MODEL_FILTER} AND c.id = ANY(%(ids)s::int[])
        """
    else:
        sql = _DOC_PASSES_OLD
    await cur.execute(sql, {"models": models, "ids": targets})
    return bool((await cur.fetchone())[0])


async def _windows(cur, enc, models, expand: bool) -> dict:
    """Полные окна dense и sparse. expand=False — прежний фильтр без семейств."""
    from rag.config import settings
    from rag.embedder import sparse_to_pgvector_literal
    from rag.retriever import _DENSE_SQL, _SPARSE_SQL

    dense_vec = enc.dense.tolist()
    sparse_lit = sparse_to_pgvector_literal(enc.sparse)
    d_sql, s_sql = (_DENSE_SQL, _SPARSE_SQL) if expand else (_OLD_DENSE, _OLD_SPARSE)

    await cur.execute(d_sql, {"dense": dense_vec, "models": models,
                              "cm": None, "limit": settings.dense_top_k})
    drows = await cur.fetchall()
    await cur.execute(s_sql, {"sparse": sparse_lit, "models": models,
                              "cm": None, "limit": settings.sparse_top_k})
    srows = await cur.fetchall()
    # Скоры обязательны, а не «на всякий случай». На кириллическом запросе
    # sparse даёт РОВНО НОЛЬ почти всему окну, и порядок среди нулей Postgres
    # выбирает произвольно. Без скоров эти строки неотличимы от найденного,
    # и слияние принимает 49 случайных чанков за свидетельства.
    return {
        "dense": [r[0] for r in drows],
        "sparse": [r[0] for r in srows],
        "dense_scores": [float(r[1]) for r in drows],
        "sparse_scores": [float(r[1]) for r in srows],
    }


async def _run(out: Path) -> None:
    from rag.config import settings
    from rag.embedder import Embedder, MAX_LEN_QUERY

    items = yaml.safe_load(Path("eval/golden_set.yaml").read_text())
    scored = [i for i in items if i.get("target_chunk")]
    try:
        from scripts.ru_en_probe import EN_PROBE
    except ImportError:
        EN_PROBE = {}

    emb = Embedder()
    ARMS = [("nofilter", None, True),
            ("m3512B", ["3512B"], True),
            ("m3512B_noexp", ["3512B"], False)]

    payload = {
        "meta": {
            "dense_top_k": settings.dense_top_k,
            "sparse_top_k": settings.sparse_top_k,
            "rrf_k": settings.rrf_k,
            "rrf_top_k": settings.rrf_top_k,
            "arms": [a[0] for a in ARMS],
        },
        "questions": {},
    }

    async with await psycopg.AsyncConnection.connect(settings.db_dsn) as conn:
        async with conn.cursor() as cur:
            for n, item in enumerate(scored, 1):
                qid = item["id"]
                click.echo(f"  [{n}/{len(scored)}] {qid}…", err=True)
                targets = await _targets(cur, item["target_chunk"])
                enc = emb.encode_one(item["question"], max_length=MAX_LEN_QUERY)

                rec = {
                    "question": item["question"],
                    "category": item.get("category", "other"),
                    "targets": targets,
                    "arms": {},
                }
                for arm, models, expand in ARMS:
                    rec["arms"][arm] = await _windows(cur, enc, models, expand)
                    rec["arms"][arm]["target_reachable"] = await _reachable(
                        cur, models, targets, expand)

                if qid in EN_PROBE:
                    en = EN_PROBE[qid]
                    enc_en = emb.encode_one(en, max_length=MAX_LEN_QUERY)
                    rec["en_question"] = en
                    rec["en_arms"] = {}
                    for arm, models, expand in ARMS[:2]:
                        rec["en_arms"][arm] = await _windows(
                            cur, enc_en, models, expand)

                payload["questions"][qid] = rec

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    click.echo(f"\nЗаписано: {out}  ({out.stat().st_size // 1024} КБ, "
               f"{len(payload['questions'])} вопросов)", err=True)


@click.command()
@click.option("--out", default="eval_ranks.json", show_default=True)
def main(out):
    asyncio.run(_run(Path(out)))


if __name__ == "__main__":
    sys.exit(main())
