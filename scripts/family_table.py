#!/usr/bin/env python3
"""
Таблица семейств моделей, ВЫВЕДЕННАЯ ИЗ КОРПУСА. Ничего не правит, только читает.

Смотреть после каждой индексации нового мануала: таблица нигде не записана
константой, она пересобирается из данных, и единственный способ убедиться,
что новый документ разложился правильно, — сверить вывод с его титульной
страницей. Строка «3500B → 3508B, 3512B, 3516B» обязана совпадать с тем,
что напечатано над серийными префиксами.

Определения (те же, что в rag/retriever.py):
  семейный токен — токен в разметке чанка, отсутствующий в списке моделей
                   СВОЕГО документа;
  покрытие       — модели того же документа с тем же поколением, где
                   поколение есть хвост после ведущих цифр (3500B → «B»).

  make families
  python3 -m scripts.family_table --models 3512B
"""
from __future__ import annotations

import asyncio
import sys

import click
import psycopg

# Общий пролог: какие токены являются именами семейств и что они покрывают.
_FAMILIES = """
WITH fam AS (
    SELECT DISTINCT unnest(c.applicable_models) AS token, c.doc_id
    FROM chunks c
), families AS (
    SELECT f.token, f.doc_id
    FROM fam f
    JOIN documents d ON d.id = f.doc_id
    WHERE NOT (f.token = ANY(d.applicable_models))
)
"""

_TABLE_SQL = _FAMILIES + """
SELECT d.filename,
       fa.token,
       regexp_replace(fa.token, '^[0-9]+', '') AS generation,
       array_agg(DISTINCT m ORDER BY m) AS covers
FROM families fa
JOIN documents d ON d.id = fa.doc_id
CROSS JOIN LATERAL unnest(d.applicable_models) AS m
WHERE regexp_replace(m, '^[0-9]+', '')
    = regexp_replace(fa.token, '^[0-9]+', '')
GROUP BY 1, 2, 3
ORDER BY 1, 2
"""

# Семейный токен, не покрывающий ни одной модели своего документа: либо
# опечатка в разметке, либо поколение, которого в документе нет. Такой чанк
# недостижим ни для одного запроса — это дефект, а не настройка.
_ORPHAN_SQL = _FAMILIES + """
SELECT d.filename, fa.token, count(DISTINCT c.id)
FROM families fa
JOIN documents d ON d.id = fa.doc_id
JOIN chunks c ON c.doc_id = fa.doc_id AND fa.token = ANY(c.applicable_models)
WHERE NOT EXISTS (
    SELECT 1 FROM unnest(d.applicable_models) AS m
    WHERE regexp_replace(m, '^[0-9]+', '')
        = regexp_replace(fa.token, '^[0-9]+', '')
)
GROUP BY 1, 2 ORDER BY 1, 2
"""

_REACH_SQL = """
SELECT c.id, c.heading, c.applicable_models, c.page_start, c.page_end
FROM chunks c JOIN documents d ON d.id = c.doc_id
WHERE (CASE WHEN cardinality(c.applicable_models) > 0
            THEN c.applicable_models ELSE d.applicable_models END) && %(wide)s::text[]
  AND NOT ((CASE WHEN cardinality(c.applicable_models) > 0
            THEN c.applicable_models ELSE d.applicable_models END) && %(narrow)s::text[])
ORDER BY c.id
"""


async def _run(models: list[str] | None) -> int:
    from rag.config import settings
    from rag.retriever import _FAMILY_CTE

    expand_sql = _FAMILY_CTE + "SELECT models FROM asked"
    async with await psycopg.AsyncConnection.connect(settings.db_dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(_TABLE_SQL)
            rows = await cur.fetchall()
            click.echo("\nТАБЛИЦА СЕМЕЙСТВ (выведена из корпуса, не задана константой)")
            click.echo("-" * 78)
            if not rows:
                click.echo("  семейных токенов в корпусе нет")
            for filename, token, generation, covers in rows:
                click.echo(f"  {filename:<24}{token:<10}поколение {generation:<6}"
                           f"→ {', '.join(covers)}")

            await cur.execute(_ORPHAN_SQL)
            orphans = await cur.fetchall()
            if orphans:
                click.secho("\nСЕМЕЙСТВА БЕЗ ЕДИНОЙ МОДЕЛИ В СВОЁМ ДОКУМЕНТЕ", fg="red")
                click.secho("чанки с такой разметкой недостижимы ни для одного "
                            "запроса — это дефект разметки", fg="red")
                for filename, token, n in orphans:
                    click.secho(f"  {filename:<24}{token:<10}{n} чанк(ов)", fg="red")

            if not models:
                click.echo()
                return 1 if orphans else 0

            await cur.execute(expand_sql, {"models": models})
            wide = (await cur.fetchone())[0]
            click.echo(f"\nРАСКРЫТИЕ ЗАПРОСА  {', '.join(models)}  →  {', '.join(wide)}")
            await cur.execute(_REACH_SQL, {"wide": wide, "narrow": models})
            gained = await cur.fetchall()
            click.echo(f"Становится достижимо чанков: {len(gained)}")
            for cid, heading, am, p0, p1 in gained:
                pages = f"стр. {p0}" if p0 == p1 else f"стр. {p0}–{p1}"
                click.echo(f"  id={cid:<6}{pages:<14}{str(am):<22}{heading[:46]}")
            click.echo()
    return 1 if orphans else 0


@click.command()
@click.option("--models", "-m", default=None,
              help="Показать, что раскрытие даёт этому запросу: 3512B")
def main(models):
    model_list = [m.strip() for m in models.split(",")] if models else None
    raise SystemExit(asyncio.run(_run(model_list)))


if __name__ == "__main__":
    sys.exit(main())
