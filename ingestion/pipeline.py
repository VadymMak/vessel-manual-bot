"""Пайплайн индексации одного PDF: PDF → элементы → чанки → JSON."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import fitz

from .chunker import Chunk, build_chunks
from .extractor import HEADER_CUTOFF, detect_heading_profile, extract_page

RE_SECTION = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Section)\b")

# Титульная страница Caterpillar: по строке на исполнение, «PREFIX 1-UP (…)».
# В скобках бывает ДВА разных содержимого, и это не вариация формата, а разные
# документы:
#
#   SEBU7844-37 (3500B/3500C):  S2D 1-UP (3508B)      — в скобках МОДЕЛЬ
#   SEBU8118-06 (C18 GenSet):   CYG 1-UP (Engine)     — в скобках РОЛЬ узла
#                               MGS 1-UP (Generator Set)
#
# У второго модель стоит не в серийных строках, а в строке продукта над ними
# («C18 Marine Generator Set»). Поэтому модели собираются из двух источников
# с приоритетом: серийные строки точнее (у SEBU7844 они дают 3508B…3516C,
# тогда как строка продукта — только общее «3500B and 3500C»), а строка
# продукта работает запасным вариантом.
#
# Якоря ^…$ по MULTILINE намеренно: без них «1-UP» поймается и в оглавлении.
RE_SERIAL_MODEL = re.compile(
    r"^([A-Z0-9]{3})\s+1-UP\s+\((3\d{3}[A-Z])\)\s*$", re.MULTILINE
)
RE_SERIAL_ANY = re.compile(
    r"^([A-Z0-9]{3})\s+1-UP\s+\([^)]+\)\s*$", re.MULTILINE
)
# Модель в строке продукта: «3516C», «C18», «C32». Ищется ТОЛЬКО в тексте выше
# первой серийной строки — ниже идут код публикации (SEBU8118-06) и дата,
# которые под шаблон подошли бы.
RE_PRODUCT_MODEL = re.compile(r"\b(3\d{3}[A-Z]|C\d{1,2})\b")


def parse_title_page(pdf_path: str | Path) -> dict:
    """
    Модели и серийные префиксы с титульной страницы.

    Это применимость документа ПО УМОЛЧАНИЮ: она действует на те чанки,
    которые не сузили её своими control_module / applicable_models,
    а таких большинство (см. migrations/002_document_applicability.sql).

    Пустые списки — законный результат: у документа без такого титула
    применимость просто не печатается, выдумывать её нельзя.
    """
    doc = fitz.open(pdf_path)
    try:
        text = doc[0].get_text("text")
    finally:
        doc.close()

    pairs = RE_SERIAL_MODEL.findall(text)
    models = sorted({m for _, m in pairs})
    prefixes = list(dict.fromkeys(RE_SERIAL_ANY.findall(text)))

    if not models:
        # Серийные строки модель не назвали (стиль C18: в скобках роль узла).
        # Берём её из строки продукта — но только из текста ВЫШЕ первой
        # серийной строки, иначе поймается код публикации.
        first = RE_SERIAL_ANY.search(text)
        head = text[: first.start()] if first else text
        models = sorted(set(RE_PRODUCT_MODEL.findall(head)))

    # Порядок префиксов сохраняем как на титуле — по нему сверяют глазами.
    # Модели сортируем: их мало, и порядок на титуле произвольный.
    return {
        # Имя файла едет вместе с метаданными НАМЕРЕННО. rag/loader.py удаляет
        # старые чанки по doc_id, и если имя задавать по умолчанию, загрузка
        # второго мануала снесёт чанки первого. Такую ошибку нельзя оставлять
        # на внимательность вызывающего.
        "filename": Path(pdf_path).name,
        "serial_prefixes": prefixes,
        "applicable_models": models,
    }


def page_sections(doc: fitz.Document) -> dict[int, str]:
    """Достать название раздела из верхнего колонтитула каждой страницы.

    Колонтитул отрезан из тела страницы, но именно он несёт
    'Maintenance Section' / 'Safety Section' — это метаданные чанка.
    """
    sections: dict[int, str] = {}
    for pno in range(doc.page_count):
        page = doc[pno]
        header = page.get_text(
            "text", clip=fitz.Rect(0, 0, page.rect.width, HEADER_CUTOFF)
        )
        m = RE_SECTION.search(header)
        if m:
            sections[pno + 1] = m.group(1)
    return sections


def ingest(pdf_path: str | Path, first_page: int = 1, last_page: int | None = None,
           on_profile=None) -> list[Chunk]:
    doc = fitz.open(pdf_path)
    last = last_page or doc.page_count

    # Профиль заголовков определяется ПО ВСЕМУ документу, а не по срезу
    # first/last: при разборе одной статьи меток icode может не оказаться
    # вовсе, и уровни поехали бы. Печатается вызывающим — человек обязан
    # сверить их с документом, автоматической проверки здесь быть не может.
    profile = detect_heading_profile(doc)
    if on_profile:
        on_profile(profile)

    elements = []
    for pno in range(first_page - 1, last):
        elements.extend(extract_page(doc[pno], pno + 1, profile))

    sections = page_sections(doc)
    doc.close()
    return build_chunks(elements, sections)


def to_json(
    chunks: list[Chunk],
    out_path: str | Path,
    document: dict | None = None,
) -> None:
    """
    Записать чанки в JSON.

    Формат изменился с плоского списка на {document, chunks}: применимости
    документа в плоском списке места не было. rag/loader.py читает оба —
    список означает «метаданных документа нет», и старые chunks.json
    грузятся как раньше.
    """
    items = []
    for c in chunks:
        d = asdict(c)
        d["has_warning"] = c.has_warning
        d["citation"] = c.citation
        items.append(d)
    payload = {"document": document or {}, "chunks": items}
    Path(out_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
