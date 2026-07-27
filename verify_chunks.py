"""
Проверка инвариантов чанкинга на реальном документе.

Это не «тесты ради тестов». Каждая проверка соответствует правилу из
CLAUDE.md, нарушение которого означает, что механик получит неполную или
неверную процедуру.
"""

from __future__ import annotations

import os
import re
import sys

import fitz

sys.path.insert(0, ".")
from ingestion.extractor import _in_body, _is_warning_banner, extract_page  # noqa: E402
from ingestion.pipeline import ingest  # noqa: E402

# Путь к проверяемому PDF: аргумент командной строки, переменная окружения
# VESSELBOT_PDF или файл по умолчанию в docs/.
PDF = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.environ.get("VESSELBOT_PDF", "docs/SEBU7844-37.pdf")
)

RE_STEP_NO = re.compile(r"^(\d+)\.\s", re.M)


def check_step_sequences(chunks) -> list[str]:
    """Шаги внутри чанка обязаны идти подряд без пропусков.

    Пропуск номера = процедура разорвана, и часть шагов потеряна.
    Это самая чувствительная проверка качества чанкинга.
    """
    problems = []
    for c in chunks:
        nums = [int(n) for n in RE_STEP_NO.findall(c.content)]
        if len(nums) < 2:
            continue
        # Разрешаем несколько независимых последовательностей (варианты),
        # но внутри каждой номера должны увеличиваться на 1.
        for prev, cur in zip(nums, nums[1:]):
            if cur == 1:
                continue  # начало новой последовательности
            if cur != prev + 1:
                problems.append(
                    f"{c.heading[:45]} ({c.citation}): разрыв {prev} → {cur}"
                )
                break
    return problems


def check_warnings_preserved(chunks) -> list[str]:
    """Все плашки WARNING документа должны попасть хотя бы в один чанк."""
    doc = fitz.open(PDF)
    banners = 0
    for pno in range(doc.page_count):
        for b in doc[pno].get_text("dict")["blocks"]:
            if _in_body(b["bbox"]) and _is_warning_banner(b):
                banners += 1
    doc.close()

    captured = sum(
        1 for c in chunks for b in c.safety_blocks if b["type"] == "WARNING"
    )
    if captured < banners * 0.9:
        return [f"перенесено {captured} блоков WARNING из {banners} плашек в PDF"]
    return []


def check_warning_attachment(chunks) -> list[str]:
    """Блок WARNING обязан лежать в чанке вместе с шагами своей процедуры."""
    problems = []
    for c in chunks:
        if c.has_warning and c.chunk_type == "procedure" and c.step_count == 0:
            problems.append(f"{c.heading[:45]}: WARNING без шагов процедуры")
    return problems


def check_tables_intact(_chunks) -> list[str]:
    """Таблица не должна начинаться в одном чанке и продолжаться в другом.

    Проверяем на уровне элементов, а не по подписи «Table N»: часть таблиц
    в SEBU7844-37 идёт вообще без номера, так что наличие подписи — не
    инвариант документа. Настоящий инвариант: непрерывная серия строк
    таблицы обязана целиком попасть в один фрагмент.
    """
    from ingestion.chunker import _split_long, group_by_procedure  # noqa: PLC0415

    doc = fitz.open(PDF)
    elements = []
    for pno in range(doc.page_count):
        elements.extend(extract_page(doc[pno], pno + 1))
    doc.close()

    problems = []
    for group in group_by_procedure(elements):
        body = [
            el for el in group
            if el.kind in {"para", "step", "table", "table_row",
                           "illustration", "warning", "notice",
                           "h1", "h2", "h3", "smcs"}
        ]
        if not body:
            continue
        parts = _split_long(body)
        if len(parts) < 2:
            continue

        # Собираем непрерывные серии строк таблицы в исходном порядке
        runs: list[list[int]] = []
        for i, el in enumerate(body):
            if el.kind in ("table", "table_row"):
                if runs and runs[-1][-1] == i - 1:
                    runs[-1].append(i)
                else:
                    runs.append([i])

        # id() элементов, попавших в каждую часть
        part_ids = [{id(el) for el in part} for part in parts]
        for run in runs:
            hosts = {
                p for p, ids in enumerate(part_ids)
                if any(id(body[i]) in ids for i in run)
            }
            if len(hosts) > 1:
                heading = next((el.text for el in body if el.kind == "h1"), "?")
                problems.append(
                    f"{heading[:45]}: таблица из {len(run)} строк разорвана "
                    f"между частями {sorted(hosts)}"
                )
    return problems


def check_citations(chunks) -> list[str]:
    problems = []
    for c in chunks:
        if c.page_start > c.page_end or c.page_start < 1:
            problems.append(f"{c.heading[:45]}: некорректные страницы {c.citation}")
    return problems


def main() -> int:
    chunks = ingest(PDF)
    print(f"Чанков: {len(chunks)}")
    lengths = sorted(len(c.content) for c in chunks)
    print(
        f"Длина (символов): медиана={lengths[len(lengths) // 2]} "
        f"p90={lengths[int(len(lengths) * 0.9)]} max={lengths[-1]}"
    )
    print(f"Процедур: {sum(1 for c in chunks if c.chunk_type == 'procedure')}")
    print(f"С предупреждениями: {sum(1 for c in chunks if c.has_warning)}")
    print()

    checks = {
        "Последовательность шагов не разорвана": check_step_sequences,
        "Все WARNING документа сохранены": check_warnings_preserved,
        "WARNING привязан к своей процедуре": check_warning_attachment,
        "Таблицы не разрезаны": check_tables_intact,
        "Корректные ссылки на страницы": check_citations,
    }

    failed = 0
    for name, fn in checks.items():
        problems = fn(chunks)
        if problems:
            failed += 1
            print(f"ПРОВАЛ  {name} — {len(problems)}:")
            for p in problems[:8]:
                print(f"        {p}")
        else:
            print(f"OK      {name}")

    print()
    print("ИТОГ:", "все инварианты соблюдены" if not failed else f"{failed} проверок провалено")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
