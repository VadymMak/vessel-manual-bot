"""
Извлечение структурированных элементов из PDF технической документации Caterpillar.

Ключевые особенности вёрстки, под которые написан модуль (проверено на SEBU7844-37):
  * Две колонки: левая x≈54, правая x≈324, страница 612pt.
    Наивный page.get_text(sort=True) перемешивает шаги двух колонок — это баг,
    а не косметика. Извлекаем колонки раздельно.
  * Границы процедур помечены ID модуля Caterpillar вида `i08246330` (7pt).
    На SEBU7844-37: 113 таких меток, каждая без исключений предшествует
    заголовку H1. Это надёжнее эвристик по размеру шрифта.
  * Плашка WARNING — ЭТО ИЗОБРАЖЕНИЕ. Слова "WARNING" в текстовом слое НЕТ.
    Детектируем по геометрии баннера (≈234×26pt), текст предупреждения —
    идущие следом блоки, целиком набранные Arial-BoldMT.
  * NOTICE — текстовый блок, обрамлённый линиями, первая строка "NOTICE".
  * Номер шага процедуры набран жирным ("8."), тело шага — обычным.

Иерархия размеров шрифта:
    15.9pt          → H1  (заголовок процедуры)
    13.9pt bold     → H2  (подраздел, напр. "Engines Equipped with ADEM II")
    12.0pt bold     → H3
    10.0pt          → основной текст
    7-8pt           → ID модуля, подписи иллюстраций, сноски
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Literal

import fitz  # PyMuPDF

# ─── Геометрия страницы ──────────────────────────────────────────────────────
PAGE_WIDTH = 612.0
COLUMN_GUTTER = 306.0  # середина; левая колонка x<306, правая x>=306
HEADER_CUTOFF = 60.0   # выше — колонтитул с номером страницы
FOOTER_CUTOFF = 750.0  # ниже — нижний колонтитул

# ─── Размеры шрифта ──────────────────────────────────────────────────────────
SIZE_H1 = 15.0
SIZE_H2 = 13.0
SIZE_H3 = 11.5
SIZE_BODY = 9.5
BOLD_FONT = "Arial-BoldMT"

# ─── Геометрия баннера WARNING ───────────────────────────────────────────────
WARN_BANNER_W = (200.0, 260.0)
WARN_BANNER_H = (18.0, 36.0)

# ─── Регулярные выражения ────────────────────────────────────────────────────
RE_ICODE = re.compile(r"^i\d{8}$")
RE_SMCS = re.compile(r"SMCS Code:\s*([\d\-;,\s]+)")
RE_STEP = re.compile(r"^(\d+)\.\s*$|^(\d+)\.\s+\S")
RE_TABLE_CAPTION = re.compile(r"^Table\s+(\d+)")
RE_ILLUSTRATION = re.compile(r"^Illustration\s+(\d+)")
RE_GRAPHIC_ID = re.compile(r"\b(g\d{8})\b")
# Номера деталей Caterpillar: 174-6854, 1U-5490, 138-8440, 9S-3263
RE_PART_NUMBER = re.compile(r"\b(\d{3}-\d{4}|\d[A-Z]-\d{4})\b")

ElementKind = Literal[
    "icode", "h1", "h2", "h3", "para", "step",
    "warning", "notice", "table", "table_row", "illustration", "smcs",
]


@dataclass
class Element:
    """Один структурный элемент страницы."""
    kind: ElementKind
    text: str
    page: int                      # 1-индексная страница PDF
    column: Literal["left", "right", "full"]
    y: float
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - отладочный вывод
        return f"<{self.kind} p{self.page} {self.column} {self.text[:50]!r}>"


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def _block_text(block: dict) -> str:
    """Собрать текст блока, схлопнув артефакты justified-вёрстки.

    PyMuPDF отдаёт растянутый текст как 'concentration   of   caustic'.
    Схлопываем множественные пробелы, но переносы строк внутри блока
    заменяем пробелом — блок это один абзац.
    """
    parts = []
    for line in block["lines"]:
        line_text = "".join(span["text"] for span in line["spans"])
        parts.append(line_text)
    text = " ".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    # Склеиваем переносы по дефису: 'prop- er procedure' → 'proper procedure'
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)
    return text.strip()


def _block_fonts(block: dict) -> set[str]:
    return {s["font"] for line in block["lines"] for s in line["spans"]}


def _block_max_size(block: dict) -> float:
    sizes = [s["size"] for line in block["lines"] for s in line["spans"] if s["text"].strip()]
    return max(sizes) if sizes else 0.0


def _is_all_bold(block: dict) -> bool:
    fonts = {s["font"] for line in block["lines"] for s in line["spans"] if s["text"].strip()}
    return bool(fonts) and fonts == {BOLD_FONT}


def _column_of(bbox: tuple[float, ...]) -> Literal["left", "right", "full"]:
    x0, _, x1, _ = bbox
    if x0 < COLUMN_GUTTER and x1 > COLUMN_GUTTER + 18:
        return "full"
    return "left" if x0 < COLUMN_GUTTER else "right"


def _is_warning_banner(block: dict) -> bool:
    """Плашка WARNING — изображение фиксированных пропорций."""
    if block.get("type") != 1:
        return False
    x0, y0, x1, y1 = block["bbox"]
    w, h = x1 - x0, y1 - y0
    return WARN_BANNER_W[0] < w < WARN_BANNER_W[1] and WARN_BANNER_H[0] < h < WARN_BANNER_H[1]


def _in_body(bbox: tuple[float, ...]) -> bool:
    return HEADER_CUTOFF <= bbox[1] <= FOOTER_CUTOFF


def _table_regions(page: fitz.Page) -> list[fitz.Rect]:
    """Найти области таблиц по нарисованной сетке.

    Таблицы в мануалах CAT нарисованы тонкими залитыми прямоугольниками
    (линиями). Кластеризуем их в области, чтобы точно понимать, какие
    текстовые блоки принадлежат таблице, а какие — соседнему абзацу.
    """
    lines = [
        d["rect"]
        for d in page.get_drawings()
        if d.get("type") == "f"
        and (d["rect"].width < 3 or d["rect"].height < 3)
        and max(d["rect"].width, d["rect"].height) > 12
    ]
    if not lines:
        return []

    # Зазор между горизонтальными линиями сетки = высота строки таблицы (~18pt).
    # Мержим по вертикали с запасом, но только если линии перекрываются по X —
    # иначе таблица в левой колонке склеится с таблицей в правой.
    ROW_GAP = 34.0

    regions: list[fitz.Rect] = []
    for rect in sorted(lines, key=lambda r: (r.y0, r.x0)):
        merged = False
        for i, reg in enumerate(regions):
            x_overlap = min(reg.x1, rect.x1) - max(reg.x0, rect.x0)
            if x_overlap > 5 and rect.y0 <= reg.y1 + ROW_GAP:
                regions[i] = reg | rect
                merged = True
                break
        if not merged:
            regions.append(fitz.Rect(rect))

    # Отбрасываем одиночные линии-разделители: у таблицы есть и высота, и ширина
    return [r for r in regions if r.height > 20 and r.width > 60]


def _strip_icode(text: str) -> tuple[str, str | None]:
    """Отделить ID модуля Caterpillar от заголовка, если они в одном блоке."""
    m = re.match(r"^(i\d{8})\s+(.*)$", text, flags=re.S)
    if m:
        return m.group(2).strip(), m.group(1)
    return text, None


def _normalize_heading(text: str) -> str:
    """Убрать артефакты переноса строк в заголовках.

    'Aftercooler Core - Inspect/ Clean/Test' → 'Aftercooler Core - Inspect/Clean/Test'
    """
    return re.sub(r"/\s+", "/", text).strip()


# ─── Классификация одного блока ──────────────────────────────────────────────

def _classify(block: dict, text: str) -> ElementKind:
    size = _block_max_size(block)
    first_span = next(
        (s for line in block["lines"] for s in line["spans"] if s["text"].strip()),
        None,
    )

    if RE_ICODE.match(text):
        return "icode"
    if text.startswith("SMCS Code:"):
        return "smcs"
    if RE_TABLE_CAPTION.match(text):
        return "table"
    if RE_ILLUSTRATION.match(text):
        return "illustration"
    if size >= SIZE_H1:
        return "h1"
    if size >= SIZE_H2 and _is_all_bold(block):
        return "h2"
    if size >= SIZE_H3 and _is_all_bold(block):
        return "h3"
    if text.startswith("NOTICE"):
        return "notice"
    # Шаг процедуры: номер набран жирным
    if first_span and first_span["font"] == BOLD_FONT and RE_STEP.match(first_span["text"].strip() + " x"):
        return "step"
    if RE_STEP.match(text):
        return "step"
    return "para"


# ─── Извлечение страницы ─────────────────────────────────────────────────────

def extract_page(page: fitz.Page, page_no: int) -> list[Element]:
    """Извлечь элементы одной страницы в корректном порядке чтения.

    Порядок: сначала полноширинные элементы, затем левая колонка сверху вниз,
    затем правая. Это восстанавливает логику чтения двухколоночной вёрстки.
    """
    raw = page.get_text("dict")["blocks"]
    blocks = [b for b in raw if _in_body(b["bbox"])]
    table_regions = _table_regions(page)

    # Индексы блоков-баннеров WARNING
    banner_ys: dict[str, list[float]] = {"left": [], "right": [], "full": []}
    for b in blocks:
        if _is_warning_banner(b):
            banner_ys[_column_of(b["bbox"])].append(b["bbox"][3])

    text_blocks = [b for b in blocks if b.get("type") == 0 and _block_text(b)]

    # Группируем по колонкам и сортируем по вертикали
    by_column: dict[str, list[dict]] = {"full": [], "left": [], "right": []}
    for b in text_blocks:
        by_column[_column_of(b["bbox"])].append(b)
    for col in by_column:
        by_column[col].sort(key=lambda b: b["bbox"][1])

    elements: list[Element] = []
    for col in ("full", "left", "right"):
        col_blocks = by_column[col]
        # Пометить блоки, идущие сразу под баннером WARNING
        warning_open = False
        for b in col_blocks:
            text = _block_text(b)
            y_top = b["bbox"][1]

            # Начало блока WARNING: блок расположен сразу под баннером
            starts_warning = any(
                0 <= y_top - banner_bottom < 12 for banner_bottom in banner_ys[col]
            )
            if starts_warning:
                warning_open = True

            if warning_open:
                if _is_all_bold(b) and _block_max_size(b) < SIZE_H3:
                    elements.append(Element("warning", text, page_no, col, y_top))
                    continue
                warning_open = False  # текст перестал быть жирным — предупреждение кончилось

            kind = _classify(b, text)
            meta: dict = {}

            # Текст внутри нарисованной сетки — строка таблицы, а не абзац
            block_rect = fitz.Rect(b["bbox"])
            if kind in ("para", "step"):
                for region in table_regions:
                    if region.intersects(block_rect) and block_rect.get_area():
                        overlap = (region & block_rect).get_area() / block_rect.get_area()
                        if overlap > 0.6:
                            kind = "table_row"
                            break

            if kind in ("h1", "h2", "h3"):
                text, icode = _strip_icode(text)
                text = _normalize_heading(text)
                if icode:
                    elements.append(Element("icode", icode, page_no, col, y_top))

            if kind == "smcs":
                m = RE_SMCS.search(text)
                if m:
                    meta["codes"] = [c.strip() for c in re.split(r"[;,]", m.group(1)) if c.strip()]
            if kind == "illustration":
                gid = RE_GRAPHIC_ID.search(text)
                if gid:
                    meta["graphic_id"] = gid.group(1)
            parts = RE_PART_NUMBER.findall(text)
            if parts:
                meta["part_numbers"] = sorted(set(parts))

            elements.append(Element(kind, text, page_no, col, y_top, meta))

    return elements


def extract_document(pdf_path: str, first_page: int = 1, last_page: int | None = None) -> Iterator[Element]:
    """Извлечь элементы всего документа в порядке чтения."""
    doc = fitz.open(pdf_path)
    last = last_page or doc.page_count
    for pno in range(first_page - 1, last):
        yield from extract_page(doc[pno], pno + 1)
    doc.close()
