"""
CLI для RAG-запросов (этап 2 — фронтенда ещё нет).

Использование:
  python3 -m rag.cli "как очистить сердцевину охладителя"
  python3 -m rag.cli "как проверить air shutoff на ADEM III" --cm "ADEM III"
  python3 -m rag.cli "промывка DEAC" --models "3516B,3516C"
  python3 -m rag.cli --load                          # загрузить chunks.json в БД
"""
from __future__ import annotations

import asyncio
import logging
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
    stream=sys.stderr,
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("question", required=False)
@click.option("--models", "-m", default=None,
              help="Модели через запятую: '3516B,3516C'")
@click.option("--cm", "--control-module", "control_module", default=None,
              help="'ADEM II' или 'ADEM III'")
@click.option("--top-n", default=None, type=int,
              help="Кол-во финальных чанков (по умолчанию из config)")
@click.option("--load", is_flag=True,
              help="Загрузить chunks.json в БД (нужна БД + OpenAI key)")
@click.option("--no-stream", is_flag=True,
              help="Собрать полный ответ перед выводом")
def main(
    question: str | None,
    models: str | None,
    control_module: str | None,
    top_n: int | None,
    load: bool,
    no_stream: bool,
) -> None:
    if load:
        asyncio.run(_cmd_load())
        return

    if not question:
        click.echo("Укажи вопрос или --load. Используй -h для справки.", err=True)
        raise SystemExit(1)

    asyncio.run(_cmd_query(question, models, control_module, top_n, no_stream))


async def _cmd_load() -> None:
    from .loader import load
    click.secho("Загружаю chunks.json в PostgreSQL…", fg="cyan", err=True)
    await load()
    click.secho("Готово.", fg="green", err=True)


async def _cmd_query(
    question: str,
    models_str: str | None,
    control_module: str | None,
    top_n: int | None,
    no_stream: bool,
) -> None:
    from .config import settings
    from .retriever import retrieve
    from .reranker import Reranker
    from .generator import generate
    from .verifier import verify

    engine_models = [m.strip() for m in models_str.split(",")] if models_str else None
    rerank_n = top_n or settings.rerank_top_n

    # ── Retrieval ──────────────────────────────────────────────────────────────
    _h("Retrieval")
    candidates = await retrieve(question, models=engine_models, control_module=control_module)
    click.secho(f"Найдено {len(candidates)} кандидатов после RRF", fg="cyan", err=True)
    for i, c in enumerate(candidates[:5]):
        click.echo(
            f"  [{i+1}] rrf={c.rrf_score:.4f}  {c.heading}  ({c.citation})",
            err=True,
        )
    if len(candidates) > 5:
        click.echo(f"  … ещё {len(candidates)-5}", err=True)

    # ── Reranking ──────────────────────────────────────────────────────────────
    _h("Reranking")
    top_chunks = Reranker().rerank(question, candidates, top_n=rerank_n)
    click.secho(f"Топ {len(top_chunks)} после реранкинга:", fg="cyan", err=True)
    for i, c in enumerate(top_chunks):
        click.echo(f"  [{i+1}] {c.heading}  ({c.citation})", err=True)

    # ── Answer ─────────────────────────────────────────────────────────────────
    _h("Answer")
    answer_parts: list[str] = []
    async for token in generate(question, top_chunks):
        answer_parts.append(token)
        if not no_stream:
            print(token, end="", flush=True)

    if no_stream:
        print("".join(answer_parts))
    else:
        print()  # перевод строки после стриминга

    full_answer = "".join(answer_parts)

    # ── Grounding check ────────────────────────────────────────────────────────
    _h("Grounding check")
    vr = verify(full_answer, top_chunks)
    if vr.ok:
        click.secho("✓ Все числа и номера деталей подтверждены в контексте.", fg="green", err=True)
    else:
        click.secho(
            f"✗ Неподтверждённые значения ({len(vr.unverified)}) — ответ заблокирован:",
            fg="red", err=True,
        )
        for v in vr.unverified:
            click.echo(f"  · {v}", err=True)
        click.secho("  → Требуется перегенерация.", fg="red", err=True)


def _h(title: str) -> None:
    click.secho(f"\n── {title} {'─' * max(0, 50 - len(title))}", fg="cyan", err=True)


if __name__ == "__main__":
    main()
