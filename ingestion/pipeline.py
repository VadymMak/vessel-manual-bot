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


def to_json(chunks: list[Chunk], out_path: str | Path) -> None:
    payload = []
    for c in chunks:
        d = asdict(c)
        d["has_warning"] = c.has_warning
        d["citation"] = c.citation
        payload.append(d)
    Path(out_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
