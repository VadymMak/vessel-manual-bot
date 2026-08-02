"""
Ретривальная оценка golden set. ДЕТЕРМИНИРОВАННАЯ, БЕЗ OpenAI.

Зачем отдельно от make eval. Поиск детерминирован, генерация — нет: замер
2026-08-02 показал, что одна и та же конфигурация даёт на gs008 14/15, 9/15
и 1/15 при побитово том же контексте, потому что дрейфует сборка модели
у провайдера (temperature=0.0 стоит с самого начала). Одним инструментом
их мерить нельзя — шум генерации топит сигнал поиска. Здесь генерации нет
вообще: эмбеддинг запроса, Neon, реранкер. Одного прогона достаточно,
и результат точный.

Целевой чанк задаётся ПРЕДИКАТОМ ПО СОДЕРЖИМОМУ (поле target_chunk
в golden_set.yaml), а не id: id меняются при переиндексации.

Использование:
  make eval-retrieval
  make eval-retrieval M="3512B"
  python3 -m eval.retrieval --ids gs007,gs008 --verbose
"""
from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import click
import psycopg
import yaml

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

# Предикат в SQL собирается из полей target_chunk. Порядок фиксирован,
# чтобы параметры и условия не разъехались.
_TARGET_SQL = """
SELECT c.id
FROM chunks c
JOIN documents d ON d.id = c.doc_id
WHERE (%(doc)s::text IS NULL OR d.filename = %(doc)s)
  AND (%(heading)s::text IS NULL OR c.heading LIKE %(heading_like)s)
  AND (%(content)s::text IS NULL OR c.content LIKE %(content_like)s)
  AND (%(cm)s::text IS NULL OR c.control_module = %(cm)s)
ORDER BY c.id
"""


@dataclass
class Row:
    id: str
    category: str
    question: str
    target_ids: list[int] = field(default_factory=list)
    dense: int | None = None
    sparse: int | None = None
    rrf: int | None = None
    rerank: int | None = None
    target_logit: float | None = None
    top1_logit: float | None = None
    top1_id: int | None = None
    error: str = ""

    @property
    def gap(self) -> float | None:
        """Разрыв логитов «цель минус первое место». 0.0 — цель и есть первая."""
        if self.target_logit is None or self.top1_logit is None:
            return None
        return self.target_logit - self.top1_logit


async def _target_ids(spec: dict) -> list[int]:
    from rag.config import settings

    heading = spec.get("heading_startswith")
    content = spec.get("content_contains")
    params = {
        "doc": spec.get("doc"),
        "heading": heading,
        "heading_like": f"{heading}%" if heading else None,
        "content": content,
        "content_like": f"%{content}%" if content else None,
        "cm": spec.get("control_module"),
    }
    async with await psycopg.AsyncConnection.connect(settings.db_dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(_TARGET_SQL, params)
            return [r[0] for r in await cur.fetchall()]


async def _branch_ranks(query: str, models: list[str] | None, targets: set[int]):
    """
    Ранги цели в ПОЛНЫХ списках dense / sparse / RRF, без обрезки до rrf_top_k.

    Полные списки нужны, чтобы отличать «цель на 26-м месте RRF» от «цели нет»:
    первое лечится порогом отсечения, второе — эмбеддингом.
    SQL-константы и функция слияния берутся из rag.retriever как есть,
    поэтому цифры здесь — те же, что видит боевой запрос.
    """
    from rag.config import settings
    from rag.embedder import Embedder, MAX_LEN_QUERY, sparse_to_pgvector_literal
    from rag.retriever import _DENSE_SQL, _SPARSE_SQL, _rrf

    enc = Embedder().encode_one(query, max_length=MAX_LEN_QUERY)
    dense_vec = enc.dense.tolist()
    sparse_lit = sparse_to_pgvector_literal(enc.sparse)

    async with await psycopg.AsyncConnection.connect(settings.db_dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                _DENSE_SQL,
                (dense_vec, models, models, None, None, dense_vec, settings.dense_top_k),
            )
            dense = [r[0] for r in await cur.fetchall()]
            await cur.execute(
                _SPARSE_SQL,
                (sparse_lit, models, models, None, None, sparse_lit, settings.sparse_top_k),
            )
            sparse = [r[0] for r in await cur.fetchall()]

    fused = [cid for cid, _ in _rrf([dense, sparse], k=settings.rrf_k)]

    def best(lst: list[int]) -> int | None:
        hits = [i + 1 for i, cid in enumerate(lst) if cid in targets]
        return min(hits) if hits else None

    return best(dense), best(sparse), best(fused)


async def _run_one(item: dict, models: list[str] | None) -> Row:
    from rag.config import settings
    from rag.retriever import retrieve
    from rag.reranker import Reranker

    row = Row(id=item["id"], category=item.get("category", "other"),
              question=item["question"])
    try:
        row.target_ids = await _target_ids(item["target_chunk"])
        if not row.target_ids:
            row.error = "предикат target_chunk не совпал ни с одним чанком"
            return row
        targets = set(row.target_ids)

        row.dense, row.sparse, row.rrf = await _branch_ranks(
            item["question"], models, targets
        )

        candidates = await retrieve(item["question"], models=models)
        if not candidates:
            row.error = "кандидатов нет"
            return row
        logits, _ = Reranker().logits(item["question"], candidates)
        ranked = sorted(zip(logits, candidates), key=lambda x: -x[0])

        row.top1_logit, top1 = ranked[0][0], ranked[0][1]
        row.top1_id = top1.id
        for pos, (lg, c) in enumerate(ranked, start=1):
            if c.id in targets:
                row.rerank, row.target_logit = pos, lg
                break
    except Exception as exc:  # noqa: BLE001
        row.error = str(exc)
    return row


async def _variant_gaps(items: list[dict], models: list[str] | None) -> dict[str, tuple]:
    """
    Для correct_variant — разрыв логитов между ПРАВИЛЬНЫМ и КОНКУРИРУЮЩИМ
    вариантом одной процедуры.

    Конкурент — чанк с тем же heading_startswith, но другим control_module.
    Отрицательный разрыв означает, что неверный вариант впереди: именно этот
    случай наблюдался на gs007 (ADEM II +5.3666 против ADEM III +4.7471).
    Это не промах поиска и не дефект промпта — реранкер плохо различает два
    исполнения одной процедуры, и двигать надо именно это число.
    """
    from rag.config import settings
    from rag.retriever import retrieve
    from rag.reranker import Reranker

    out: dict[str, tuple] = {}
    for item in items:
        spec = item["target_chunk"]
        # Конкурент ищется по СОСЕДЯМ ПО ЗАГОЛОВКУ, а не по control_module:
        # исполнения различаются им только у Air Shutoff. У Coolant - Change
        # варианты Cat ELC и Cat DEAC живут под одним заголовком и
        # различаются содержимым, а спутать их так же опасно — правило 5
        # CLAUDE.md. Поэтому берём всё с тем же doc + heading, кроме цели.
        neighbour_spec = {k: v for k, v in spec.items()
                          if k in ("doc", "heading_startswith")}
        right = set(await _target_ids(spec))
        rivals = set(await _target_ids(neighbour_spec)) - right
        if not rivals:
            continue

        candidates = await retrieve(item["question"], models=models)
        if not candidates:
            continue
        logits, _ = Reranker().logits(item["question"], candidates)
        by_id = {c.id: lg for lg, c in zip(logits, candidates)}
        head_by_id = {c.id: c.heading for c in candidates}
        right_seen = {i: by_id[i] for i in right if i in by_id}
        rival_seen = {i: by_id[i] for i in rivals if i in by_id}
        if not right_seen or not rival_seen:
            continue
        r_id = max(right_seen, key=right_seen.get)
        v_id = max(rival_seen, key=rival_seen.get)
        label = spec.get("control_module") or head_by_id.get(r_id, "")[:22]
        out[item["id"]] = (
            r_id, right_seen[r_id], v_id, rival_seen[v_id],
            right_seen[r_id] - rival_seen[v_id], label,
        )
    return out


def _print(rows: list[Row], gaps: dict[str, tuple], verbose: bool) -> None:
    from rag.config import settings

    n = settings.rerank_top_n
    click.echo("\n" + "=" * 108)
    click.echo("РЕТРИВАЛЬНАЯ ОЦЕНКА — детерминированная, без генерации")
    click.echo("=" * 108)
    click.echo(
        f"{'id':<8}{'категория':<18}{'dense':>7}{'sparse':>7}{'RRF':>6}"
        f"{'ранк':>6}{'логит цели':>12}{'логит #1':>11}{'разрыв':>9}  цель"
    )
    for r in rows:
        if r.error:
            click.secho(f"{r.id:<8}{r.category:<18}  ОШИБКА: {r.error}", fg="red")
            continue
        f = lambda v: "—" if v is None else str(v)  # noqa: E731
        gap = r.gap
        gap_s = "—" if gap is None else f"{gap:+.3f}"
        color = None if r.rerank == 1 else ("yellow" if (r.rerank or 99) <= n else "red")
        click.secho(
            f"{r.id:<8}{r.category:<18}{f(r.dense):>7}{f(r.sparse):>7}{f(r.rrf):>6}"
            f"{f(r.rerank):>6}"
            f"{'—' if r.target_logit is None else f'{r.target_logit:+.3f}':>12}"
            f"{'—' if r.top1_logit is None else f'{r.top1_logit:+.3f}':>11}"
            f"{gap_s:>9}  {r.target_ids}",
            fg=color,
        )
        if verbose:
            click.echo(f"         «{r.question[:90]}»")

    ok = [r for r in rows if not r.error and r.rerank]
    first = sum(1 for r in ok if r.rerank == 1)
    in_top = sum(1 for r in ok if r.rerank <= n)
    missing = [r.id for r in rows if not r.error and r.rerank is None]
    mrr = sum(1.0 / r.rerank for r in ok) / len(rows) if rows else 0.0

    click.echo("=" * 108)
    click.secho(f"Цель на первом месте после реранкинга: {first}/{len(rows)}", bold=True)
    click.echo(f"Цель в топ-{n}:                          {in_top}/{len(rows)}")
    click.secho(f"MRR:                                   {mrr:.4f}", bold=True)
    if missing:
        click.secho(f"Цель не дошла до реранкера: {', '.join(missing)}", fg="red")

    if gaps:
        click.echo("\n" + "-" * 108)
        click.echo("РАЗРЫВ ЛОГИТОВ МЕЖДУ ВАРИАНТАМИ ОДНОЙ ПРОЦЕДУРЫ (correct_variant)")
        click.echo("отрицательный разрыв = неверный вариант впереди")
        click.echo(f"{'id':<8}{'верный':<24}{'цель':>7}{'логит цели':>12}"
                   f"{'конкурент':>11}{'логит конк.':>13}{'разрыв':>10}")
        for qid, (r_id, r_lg, v_id, v_lg, d, label) in sorted(gaps.items()):
            click.secho(
                f"{qid:<8}{label:<24}{r_id:>7}{r_lg:>+12.4f}"
                f"{v_id:>11}{v_lg:>+13.4f}{d:>+10.4f}",
                fg="green" if d > 0 else "red",
            )
    click.echo()


@click.command()
@click.option("--ids", default=None, help="Список ID через запятую: gs007,gs008")
@click.option("--category", "-c", default=None, help="Только одна категория")
@click.option("--models", "-m", "models", default=None,
              help="Фильтр по модели двигателя, идёт в SQL WHERE (правило 7)")
@click.option("--verbose", "-v", is_flag=True, help="Печатать текст вопросов")
@click.option("--golden-set", default="eval/golden_set.yaml", show_default=True)
def main(ids, category, models, verbose, golden_set):
    items: list[dict] = yaml.safe_load(Path(golden_set).read_text())
    total = len(items)

    # honest_refusal целевого чанка не имеет по определению: правильного
    # ответа в мануале нет. Молча их пропустить нельзя — счёт должен сходиться.
    scored = [i for i in items if i.get("target_chunk")]
    skipped = [i for i in items if not i.get("target_chunk")]

    if category:
        scored = [i for i in scored if i.get("category") == category]
    if ids:
        want = {s.strip() for s in ids.split(",")}
        scored = [i for i in scored if i["id"] in want]

    if not scored:
        click.echo("Нет вопросов с target_chunk после фильтрации.", err=True)
        raise SystemExit(1)

    model_list = [m.strip() for m in models.split(",")] if models else None
    click.echo(
        f"Вопросов в golden set: {total}. "
        f"С разметкой target_chunk: {len(scored)}. "
        f"Без разметки (только для генеративной оценки): {len(skipped)}"
        + (f" — {', '.join(i['id'] for i in skipped)}" if skipped else ""),
        err=True,
    )
    if model_list:
        click.echo(f"Фильтр по модели: {', '.join(model_list)}", err=True)

    rows = []
    for i, item in enumerate(scored, 1):
        click.echo(f"  [{i}/{len(scored)}] {item['id']}…", err=True)
        rows.append(asyncio.run(_run_one(item, model_list)))

    gaps = asyncio.run(_variant_gaps(
        [i for i in scored
         if i.get("category") == "correct_variant"
         or i["target_chunk"].get("control_module")],
        model_list,
    ))
    _print(rows, gaps, verbose)


if __name__ == "__main__":
    main()
