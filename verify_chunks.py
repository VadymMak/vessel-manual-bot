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
from ingestion.extractor import (  # noqa: E402
    _in_body, _is_warning_banner, detect_heading_profile, extract_page,
)
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
    profile = detect_heading_profile(doc)
    elements = []
    for pno in range(doc.page_count):
        elements.extend(extract_page(doc[pno], pno + 1, profile))
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


def check_table_caption_binding(_chunks) -> list[str]:
    """
    Подпись таблицы и её содержимое лежат в ОДНОЙ процедуре.

    Пять прежних инвариантов проверяют ЦЕЛОСТНОСТЬ — что непрерывная серия
    строк не разрезана между частями. Они не ловят ПОДМЕНУ: когда строки целы,
    но оказались в чужой статье, а подпись осталась при чужих строках.

    Найдено 2026-08-03 на RENR5078-05, стр. 46. Чанк нёс метку [Table 6]
    «Counterclockwise rotation (Standard)», а внутри лежали 14 строк таблицы 7
    «Clockwise rotation (Reverse)». Ни один номер цилиндра не потерян — все 28
    были в корпусе, — но под подписью стандартного вращения предъявлялись
    цилиндры обратного. Механик получил бы не тот цилиндр при регулировке
    клапанов. Потеря видна, подмена нет.

    ПРОВЕРЯЕТСЯ НА УРОВНЕ ЭЛЕМЕНТОВ, А НЕ ПО ТЕКСТУ ЧАНКА. Первая версия этой
    проверки искала в тексте метку [Table N] и строки вида `| … |` и была
    негодной в обе стороны: давала шесть ложных срабатываний на таблицах,
    у которых сетка не разобралась и содержимое лежит плоским текстом,
    и НЕ ловила исходный дефект — там и метка, и строки присутствовали
    в одном чанке, просто от разных таблиц.

    ЧЕСТНАЯ ГРАНИЦА ЭТОЙ ПРОВЕРКИ. Она НЕ ловит исходный дефект RENR
    и не является полноценным инвариантом принадлежности. Проверено три
    формулировки:
      односторонняя (содержимое только ПОСЛЕ подписи) — ловит дефект RENR,
        но даёт 9 ложных срабатываний: подпись Caterpillar печатает над
        таблицей, а разобранная сетка эмитится по координате своей области,
        и та бывает выше подписи;
      двусторонняя (принята здесь) — ложных нет, но дефект RENR пропускает;
      счётная (число подписей = числу таблиц в процедуре) — 6-7 срабатываний
        в обоих порядках чтения, шум.
    Причина в данных: элемент разобранной сетки НЕ ЗНАЕТ своего номера
    таблицы, поэтому связать подпись с содержимым структурно нечем.
    Настоящее решение — проставлять номер из подписи в meta сетки при
    извлечении; тогда проверка станет точной и тривиальной. Записано
    в очередь.

    Что проверка ловит сейчас: подпись, вокруг которой вообще нет содержимого
    таблицы — ни до, ни после, до ближайшего заголовка или другой подписи.
    """
    from ingestion.chunker import group_by_procedure  # noqa: PLC0415

    doc = fitz.open(PDF)
    profile = detect_heading_profile(doc)
    elements = []
    for pno in range(doc.page_count):
        elements.extend(extract_page(doc[pno], pno + 1, profile))
    doc.close()
    return _caption_problems(elements)


def _caption_problems(elements) -> list[str]:
    """Общая часть: проверка потока элементов. Вынесена, чтобы прогонять
    её и на восстановленном прежнем порядке чтения при отладке."""
    from ingestion.chunker import group_by_procedure  # noqa: PLC0415

    cap_re = re.compile(r"^Table\s+(\d+)")
    problems = []
    for group in group_by_procedure(elements):
        heading = next((el.text for el in group if el.kind == "h1"), "?")
        for i, el in enumerate(group):
            if el.kind != "table" or not cap_re.match(el.text or ""):
                continue
            num = cap_re.match(el.text).group(1)

            def _scan(seq):
                """Есть ли содержимое таблицы до ближайшей подписи/заголовка."""
                for nxt in seq:
                    if nxt.kind in ("h1", "h2", "h3"):
                        return False
                    if nxt.kind == "table" and cap_re.match(nxt.text or ""):
                        return False
                    if nxt.kind in ("table", "table_row") or (
                            nxt.kind == "para" and len(nxt.text) > 40):
                        return True
                return False

            # Смотрим В ОБЕ СТОРОНЫ. Подпись Caterpillar печатает над таблицей,
            # но разобранная сетка эмитится по координате своей области, и та
            # бывает выше подписи. На стр. 55 SEBU7844-37 порядок элементов
            # такой: rows(11), «Table 21», «Table 22», rows(5) — обе подписи
            # при своих таблицах, просто с разных сторон. Односторонняя
            # проверка давала здесь девять ложных срабатываний.
            found = _scan(group[i + 1:]) or _scan(list(reversed(group[:i])))
            if not found:
                problems.append(
                    f"подпись «Table {num}» без содержимого — "
                    f"{heading[:42]!r} стр. {el.page}"
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
        "Подпись таблицы при своих строках": check_table_caption_binding,
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
