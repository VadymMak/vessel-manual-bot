"""
Прогон golden set через полный RAG-пайплайн.

Использование:
  python3 -m eval.run
  python3 -m eval.run --category number_accuracy
  python3 -m eval.run --id gs007,gs008    # только конкретные вопросы
  python3 -m eval.run --no-rerank         # пропустить реранкинг (быстрее)

Метрики:
  number_accuracy  — точность числовых значений с единицами
  part_number      — точность номеров деталей
  correct_variant  — правильный вариант процедуры (ADEM II/III, ELC/DEAC)
  honest_refusal   — доля корректных отказов (нет в документации)
  step_count       — правильные детали процедур
  procedure_detail — детали процедур
  overall          — общий процент прохождения
"""
from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import click
import yaml

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    id: str
    question: str
    category: str
    passed: bool
    answer: str
    missing: list[str] = field(default_factory=list)   # expected_contains не найдены
    forbidden: list[str] = field(default_factory=list)  # must_not_contain найдены
    pages_cited: list[int] = field(default_factory=list)
    error: str = ""


def _satisfied(expectation, answer_lower: str) -> bool:
    """
    Одно ожидание выполнено?

    Строка   — строгая проверка вхождения (числа, номера деталей, коды).
    Список   — «любой из вариантов»: одно и то же понятие по-русски и
               по-английски. Система отвечает на языке вопроса (правило 7
               системного промпта), поэтому требовать именно английское
               'wiring' от русского ответа — дефект теста, а не системы.
    """
    if isinstance(expectation, (list, tuple)):
        return any(str(v).lower() in answer_lower for v in expectation)
    return str(expectation).lower() in answer_lower


def _fmt_expectation(expectation) -> str:
    if isinstance(expectation, (list, tuple)):
        return " | ".join(str(v) for v in expectation) + "  (любой из вариантов)"
    return str(expectation)


def _check(answer: str, item: dict) -> EvalResult:
    answer_lower = answer.lower()

    missing = [
        _fmt_expectation(exp) for exp in item.get("expected_contains", [])
        if not _satisfied(exp, answer_lower)
    ]
    forbidden = [
        s for s in item.get("must_not_contain", [])
        if str(s).lower() in answer_lower
    ]

    # Извлекаем упомянутые страницы из ответа (ссылки [SEBU7844-37, стр. X])
    import re
    cited = [int(p) for p in re.findall(r"(?:стр\.|p\.)\s*(\d+)", answer)]

    passed = not missing and not forbidden

    return EvalResult(
        id=item["id"],
        question=item["question"],
        category=item.get("category", "other"),
        passed=passed,
        answer=answer,
        missing=missing,
        forbidden=forbidden,
        pages_cited=cited,
    )


async def _run_one(item: dict, use_rerank: bool) -> EvalResult:
    from rag.retriever import retrieve
    from rag.reranker import Reranker
    from rag.generator import generate
    from rag.config import settings

    try:
        candidates = await retrieve(item["question"])
        if use_rerank and candidates:
            top = Reranker().rerank(item["question"], candidates, top_n=settings.rerank_top_n)
        else:
            top = candidates[:settings.rerank_top_n]

        parts: list[str] = []
        async for token in generate(item["question"], top):
            parts.append(token)
        answer = "".join(parts)

    except Exception as exc:
        return EvalResult(
            id=item["id"],
            question=item["question"],
            category=item.get("category", "other"),
            passed=False,
            answer="",
            error=str(exc),
        )

    return _check(answer, item)


def _print_summary(results: list[EvalResult]) -> None:
    from collections import defaultdict

    total = len(results)
    passed = sum(1 for r in results if r.passed)

    click.echo("\n" + "=" * 70)
    click.echo(f"РЕЗУЛЬТАТ: {passed}/{total} прошло ({100*passed//total}%)")
    click.echo("=" * 70)

    # По категориям
    by_cat: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    click.echo("\nПо категориям:")
    for cat, cat_results in sorted(by_cat.items()):
        cat_passed = sum(1 for r in cat_results if r.passed)
        color = "green" if cat_passed == len(cat_results) else "yellow" if cat_passed else "red"
        click.secho(
            f"  {cat:<20} {cat_passed}/{len(cat_results)}",
            fg=color,
        )

    # Детали провалов
    failed = [r for r in results if not r.passed]
    if failed:
        click.echo(f"\nПровалы ({len(failed)}):")
        for r in failed:
            click.secho(f"\n  ✗ [{r.id}] {r.question[:60]}", fg="red")
            if r.error:
                click.echo(f"    ERROR: {r.error}")
            for m in r.missing:
                click.echo(f"    MISSING: {m!r}")
            for f in r.forbidden:
                click.echo(f"    FORBIDDEN found: {f!r}")
            if r.answer:
                click.echo(f"    Ответ (первые 200 симв.):\n    {r.answer[:200]!r}")

    click.echo()


@click.command()
@click.option("--category", "-c", default=None,
              help="Фильтр по категории: number_accuracy, part_number, ...")
@click.option("--id", "ids", default=None,
              help="Список ID через запятую: gs001,gs007")
@click.option("--no-rerank", is_flag=True,
              help="Пропустить реранкинг (быстрее, ниже качество)")
@click.option("--golden-set", default="eval/golden_set.yaml",
              show_default=True, help="Путь к файлу golden set")
def main(category: str | None, ids: str | None, no_rerank: bool, golden_set: str) -> None:
    items: list[dict] = yaml.safe_load(Path(golden_set).read_text())

    if category:
        items = [i for i in items if i.get("category") == category]
    if ids:
        id_set = {s.strip() for s in ids.split(",")}
        items = [i for i in items if i["id"] in id_set]

    if not items:
        click.echo("Нет вопросов после фильтрации.", err=True)
        raise SystemExit(1)

    click.echo(f"Запускаю {len(items)} вопросов…", err=True)

    results: list[EvalResult] = []
    for i, item in enumerate(items):
        click.echo(f"  [{i+1}/{len(items)}] {item['id']}: {item['question'][:50]}…", err=True)
        r = asyncio.run(_run_one(item, use_rerank=not no_rerank))
        results.append(r)
        status = "✓" if r.passed else "✗"
        click.secho(f"    {status}", fg="green" if r.passed else "red", err=True)

    _print_summary(results)


if __name__ == "__main__":
    main()
