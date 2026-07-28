"""Пайплайн индексации одного PDF: PDF → элементы → чанки → JSON."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import fitz

from .chunker import Chunk, build_chunks
from .extractor import HEADER_CUTOFF, extract_page

RE_SECTION = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Section)\b")

# Титульная страница Caterpillar: по строке на исполнение, строго
# «PREFIX 1-UP (MODEL)» — «S2D 1-UP (3508B)». У SEBU7844-37 таких строк 50,
# в них 50 уникальных префиксов и 6 моделей. Якоря ^…$ по MULTILINE намеренно:
# без них «1-UP» поймается и в оглавлении.
RE_SERIAL_LINE = re.compile(
    r"^([A-Z0-9]{3})\s+1-UP\s+\((3\d{3}[A-Z])\)\s*$", re.MULTILINE
)


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

    pairs = RE_SERIAL_LINE.findall(text)
    # Порядок префиксов сохраняем как на титуле — по нему сверяют глазами.
    # Модели сортируем: их мало, и порядок на титуле произвольный.
    return {
        "serial_prefixes": list(dict.fromkeys(p for p, _ in pairs)),
        "applicable_models": sorted({m for _, m in pairs}),
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


def ingest(pdf_path: str | Path, first_page: int = 1, last_page: int | None = None) -> list[Chunk]:
    doc = fitz.open(pdf_path)
    last = last_page or doc.page_count

    elements = []
    for pno in range(first_page - 1, last):
        elements.extend(extract_page(doc[pno], pno + 1))

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
