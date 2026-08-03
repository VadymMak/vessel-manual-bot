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
    идущие следом блоки, целиком набранные жирным (флаг, не имя шрифта).
  * NOTICE — текстовый блок, обрамлённый линиями, первая строка "NOTICE".
  * Номер шага процедуры набран жирным ("8."), тело шага — обычным.

Иерархия размеров шрифта НЕ ЗАШИТА: она определяется из документа функцией
detect_heading_profile() и печатается при make ingest. У SEBU7844-37 и C18
она такая (RENR того же семейства даёт другие числа — в этом и была причина):
    17.9pt          → уровень секции ("Safety Section"), тоже классифицируется h1
    15.9pt          → H1  (заголовок процедуры, подтверждён меткой icode)
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
# Константами БОЛЬШЕ НЕ ЗАДАЮТСЯ — профиль определяется из самого документа,
# см. detect_heading_profile(). Три документа семейства дают три разных набора:
#
#   SEBU7844-37, C18   секция 17.9   H1 15.9   H2 13.9   H3 12.0   тело 10.0
#   RENR5078-05        (по замеру)   H1 17.9   H2 15.8   H3 13.9
#
# Допуск: размер относится к уровню, если он не меньше уровня минус TOL.
# 0.5 подобран не на глаз — он воспроизводит прежние пороги 15.0 / 13.0 / 11.5
# на обоих проиндексированных мануалах побайтово, при том что 11.0pt
# (5-6% символов, почти не жирный) в H3 по-прежнему не попадает.
SIZE_TOL = 0.5
SIZE_BODY = 9.5

# Жирность — ПО ФЛАГУ, а не по имени шрифта. Имя меняется от документа
# к документу (Arial-BoldMT против Arial,Bold), бит 2**4 в span["flags"] нет.
BOLD_FLAG = 2 ** 4

# Метка процедуры Caterpillar. Блок, начинающийся с неё, ЕСТЬ заголовок H1
# по построению документа — это структурный якорь, а не эвристика по размеру.
RE_LEADING_ICODE = re.compile(r"^i\d{8}(\s|$)")


@dataclass(frozen=True)
class HeadingProfile:
    """Размеры уровней заголовков КОНКРЕТНОГО документа."""
    body: float
    h1: float
    h2: float
    h3: float
    h1_anchors: int          # сколько блоков с меткой icode подтвердили H1
    h1_agreement: float      # доля этих блоков, согласившихся на один размер

    def kind_of(self, size: float, is_bold: bool) -> str | None:
        """Уровень заголовка по размеру, либо None."""
        if size >= self.h1 - SIZE_TOL:
            return "h1"
        if is_bold and size >= self.h2 - SIZE_TOL:
            return "h2"
        if is_bold and size >= self.h3 - SIZE_TOL:
            return "h3"
        return None

    @property
    def h3_cutoff(self) -> float:
        """Ниже этого размера жирный блок заголовком уже не считается."""
        return self.h3 - SIZE_TOL

    def describe(self) -> str:
        return (f"профиль заголовков: тело {self.body}  H1 {self.h1}  "
                f"H2 {self.h2}  H3 {self.h3}   "
                f"(H1 подтверждён {self.h1_anchors} метками icode, "
                f"согласие {self.h1_agreement:.0%})")


def detect_heading_profile(doc: fitz.Document) -> HeadingProfile:
    """
    Определить размеры уровней заголовков из самого документа.

    ПОЧЕМУ НЕ ПРОСТО ГИСТОГРАММА «доминирующий размер — тело, три следующих
    сверху — H1/H2/H3». Потому что уровней ЧЕТЫРЕ, а не три: 17.9pt несёт
    «Safety Section» и «Product Information Section» на 17-20 страницах
    обоих мануалов. Правило «три сверху» дало бы H1=17.9, H2=15.9, H3=13.9
    и сдвинуло бы всю иерархию на уровень — 12.0pt перестал бы быть
    заголовком вовсе. Проверено на данных, не предположение.

    ПОЭТОМУ H1 БЕРЁТСЯ ОТ СТРУКТУРНОГО ЯКОРЯ. Каждая процедура Caterpillar
    помечена ID модуля вида i08246330, и метка стоит в одном блоке
    с заголовком. Размер таких блоков и есть H1 — по построению документа,
    а не по частоте. Замер: SEBU7844-37 — 112 меток, согласие 100%;
    C18 — 139 меток, согласие 100%.

    H2 и H3 — два ближайших размера СТРОГО между телом и H1, у которых
    большинство символов жирные. У обоих мануалов это 13.9 и 12.0; размер
    11.0pt (5-6% символов) отсеивается тем, что жирного в нём 0.5%.

    Размер выше H1 остаётся уровнем секции и классифицируется как h1 —
    ровно так же, как это делали прежние константы.
    """
    from collections import Counter

    anchors: Counter[float] = Counter()
    all_chars: Counter[float] = Counter()
    bold_chars: Counter[float] = Counter()

    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block or not _in_body(block["bbox"]):
                continue
            if RE_LEADING_ICODE.match(_block_text(block)):
                anchors[round(_block_max_size(block), 1)] += 1
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    size = round(span["size"], 1)
                    all_chars[size] += len(text)
                    if span["flags"] & BOLD_FLAG:
                        bold_chars[size] += len(text)

    if not all_chars:
        raise ValueError("в документе нет текстового слоя")

    body = all_chars.most_common(1)[0][0]

    if anchors:
        h1, hits = anchors.most_common(1)[0]
        n_anchors, agreement = sum(anchors.values()), hits / sum(anchors.values())
    else:
        # Документ без меток процедур. Падать нельзя, но и молчать нельзя:
        # H1 берётся как крупнейший жирный размер над телом, а нулевое
        # число якорей в логе показывает, что уровень не подтверждён.
        candidates = [s for s in all_chars
                      if s > body and bold_chars.get(s, 0) > all_chars[s] / 2]
        if not candidates:
            raise ValueError("не найдено ни одного жирного размера крупнее тела")
        h1, n_anchors, agreement = max(candidates), 0, 0.0

    # H2 и H3 — два ближайших жирных уровня строго между телом и H1.
    between = sorted(
        (s for s in all_chars
         if body < s < h1 and bold_chars.get(s, 0) > all_chars[s] / 2),
        reverse=True,
    )
    if len(between) < 2:
        raise ValueError(
            f"между телом {body} и H1 {h1} найдено уровней: {between} — "
            "ожидалось не меньше двух (H2 и H3)"
        )
    h2, h3 = between[0], between[1]
    return HeadingProfile(body=body, h1=h1, h2=h2, h3=h3,
                          h1_anchors=n_anchors, h1_agreement=agreement)

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
    """Все непустые спаны блока набраны жирным.

    Проверяется ФЛАГ, а не имя шрифта: SEBU7844-37 и C18 используют
    Arial-BoldMT, RENR — Arial,Bold, и список имён пришлось бы вести вручную.
    Бит 2**4 одинаков у всех. На обоих проиндексированных мануалах правило
    по флагу совпадает с прежним правилом по имени полностью.
    """
    spans = [s for line in block["lines"] for s in line["spans"] if s["text"].strip()]
    return bool(spans) and all(s["flags"] & BOLD_FLAG for s in spans)


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


HEADER_RULE_CUTOFF = 70.0


def _is_header_rule(rect: fitz.Rect) -> bool:
    """Отличить разделитель колонтитула от границы таблицы.

    Под верхним колонтитулом на большинстве страниц проведена горизонтальная
    линейка во всю ширину полосы (y≈62 и y≈64 — одинаково в SEBU7844-37
    и в C18 Marine Generator Set). Она пересекает ОБЕ колонки, и если рядом
    сверху страницы начинается таблица, кластеризация склеивает через эту
    линейку левую колонку с правой.

    Реальный случай: C18, стр. 140 — регион растянулся на (54…558) и проглотил
    шаги 3–5 раздела Lubrication System вместе с Table 21 из правой колонки.
    Проверка «последовательность шагов не разорвана» это поймала: разрыв 3 → 6.

    Настоящие таблицы начинаются не выше y≈92: над сеткой всегда есть подпись
    либо заголовок.
    """
    return rect.height < 3 and rect.width > 400 and rect.y0 < HEADER_RULE_CUTOFF


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
        and not _is_header_rule(d["rect"])
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


def _grid_lines(page: fitz.Page, region: fitz.Rect) -> tuple[list[fitz.Rect], list[float]]:
    """Вернуть вертикальные линии сетки и границы строк внутри области таблицы."""
    verticals: list[fitz.Rect] = []
    horizontals: list[float] = []
    # Расширяем область поиска: внешние линии рамки лежат ровно на границе
    # региона, а Rect.intersects() для соприкасающихся краёв даёт False.
    probe = fitz.Rect(region.x0 - 3, region.y0 - 3, region.x1 + 3, region.y1 + 3)
    for d in page.get_drawings():
        if d.get("type") != "f":
            continue
        r = d["rect"]
        if not probe.intersects(r):
            continue
        if r.width < 3 and r.height > 4:
            verticals.append(fitz.Rect(r))
        elif r.height < 3 and r.width > 20:
            horizontals.append(round(r.y0, 1))
    return verticals, sorted(set(horizontals))


def extract_table_rows(page: fitz.Page, region: fitz.Rect) -> list[list[str]]:
    """Разобрать таблицу на ячейки по нарисованной сетке.

    Зачем это нужно, а не просто взять текст строкой:
    строка «1U-5490 Hydrosolv 4165 19 L (5 US gal)» без разделителей колонок
    неоднозначна. На вопрос «какой номер детали у Hydrosolv 4165» модель видит
    сплошной поток, где 4165 может быть и частью названия, и номером. На
    SEBU7844-37 это давало стабильные провалы именно по номерам деталей
    при том, что нужный чанк находился корректно.

    Вертикальные линии определяются ДЛЯ КАЖДОЙ СТРОКИ отдельно: в шапке
    таблицы ячейка обычно объединена на всю ширину, и общий набор границ
    разрезал бы заголовок посередине слова.
    """
    verticals, h_lines = _grid_lines(page, region)
    if len(h_lines) < 2:
        return []

    rows: list[list[str]] = []
    for top, bottom in zip(h_lines, h_lines[1:]):
        if bottom - top < 6:  # сдвоенная линия, не строка
            continue
        mid = (top + bottom) / 2
        # Границы колонок ИМЕННО ЭТОЙ строки: внутренние берём из вертикальных
        # линий, пересекающих строку, внешние — всегда края региона.
        inner = {
            round(v.x0, 1) for v in verticals
            if v.y0 - 2 <= mid <= v.y1 + 2
            and region.x0 + 2 < v.x0 < region.x1 - 2
        }
        xs = sorted({round(region.x0, 1), *inner, round(region.x1, 1)})

        cells: list[str] = []
        for left, right in zip(xs, xs[1:]):
            clip = fitz.Rect(left + 1, top + 1, right - 1, bottom - 1)
            text = page.get_text("text", clip=clip).strip()
            text = re.sub(r"\s+", " ", text)
            cells.append(text)

        if any(cells):
            rows.append(cells)

    return rows


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

def _classify(block: dict, text: str, profile: HeadingProfile) -> ElementKind:
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
    level = profile.kind_of(size, _is_all_bold(block))
    if level:
        return level  # type: ignore[return-value]
    if text.startswith("NOTICE"):
        return "notice"
    # Шаг процедуры: номер набран жирным
    if (first_span and first_span["flags"] & BOLD_FLAG
            and RE_STEP.match(first_span["text"].strip() + " x")):
        return "step"
    if RE_STEP.match(text):
        return "step"
    return "para"


# ─── Извлечение страницы ─────────────────────────────────────────────────────

def extract_page(page: fitz.Page, page_no: int,
                 profile: HeadingProfile) -> list[Element]:
    """Извлечь элементы одной страницы в корректном порядке чтения.

    ПОРЯДОК — ПОЛОСАМИ, а не «сначала все полноширинные, потом колонки».
    Полноширинные элементы делят страницу на горизонтальные полосы; внутри
    полосы читается левая колонка сверху вниз, затем правая.

    Почему не «сначала full, потом left, потом right», как было. Такой порядок
    вырывает полноширинный элемент из места, где он стоит, и уносит в начало
    страницы. На RENR5078-05 стр. 46 это привело к подмене, а не к потере:
    строки Table 6 оказались в предыдущей процедуре «Fuel Injector Adjustment»,
    а подпись [Table 6] осталась при строках Table 7. Чанк предъявлял номера
    цилиндров ОБРАТНОГО вращения под подписью СТАНДАРТНОГО. Потеря видна,
    подмена нет.

    Почему не «сортировать всё по y». Это сломало бы нормальные двухколоночные
    страницы, где текст течёт из низа левой колонки в верх правой: сортировка
    по y перемешала бы шаги 1-2 с 8-10 — ровно тот баг, ради которого
    колоночное чтение и вводилось (правило 2 CLAUDE.md).

    Полосы сохраняют колоночный поток там, где он есть, и прерывают его
    ровно там, где страница действительно полноширинная.
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

    # ─── Порядок чтения полосами ─────────────────────────────────────────────
    # Границы полос — верхние края полноширинных блоков. Колоночный блок
    # попадает в ту полосу, в которую попадает его верхний край.
    ordered: list[tuple[str, dict]] = []
    prev_y = float("-inf")
    for full_block in by_column["full"]:
        y0 = full_block["bbox"][1]
        for col in ("left", "right"):
            ordered += [(col, b) for b in by_column[col]
                        if prev_y <= b["bbox"][1] < y0]
        ordered.append(("full", full_block))
        prev_y = y0
    for col in ("left", "right"):
        ordered += [(col, b) for b in by_column[col] if b["bbox"][1] >= prev_y]

    elements: list[Element] = []
    emitted_tables: set[int] = set()
    # warning_open жил в области видимости одной колонки. Теперь колонка
    # меняется внутри страницы, и состояние обязано сбрасываться на каждой
    # смене — иначе жирный текст из правой колонки прилипнет к баннеру левой.
    prev_col: str | None = None
    warning_open = False
    for col, b in ordered:
        if col != prev_col:
            warning_open = False
            prev_col = col
        text = _block_text(b)
        y_top = b["bbox"][1]

        # Начало блока WARNING: блок расположен сразу под баннером
        starts_warning = any(
            0 <= y_top - banner_bottom < 12 for banner_bottom in banner_ys[col]
        )
        if starts_warning:
            warning_open = True

        if warning_open:
            if _is_all_bold(b) and _block_max_size(b) < profile.h3_cutoff:
                elements.append(Element("warning", text, page_no, col, y_top))
                continue
            warning_open = False  # текст перестал быть жирным — предупреждение кончилось

        kind = _classify(b, text, profile)
        meta: dict = {}

        # Текст внутри нарисованной сетки принадлежит таблице.
        # Таблицу разбираем ЦЕЛИКОМ по ячейкам при первом попавшемся
        # внутреннем блоке, остальные блоки этой области пропускаем:
        # плоская строка «1U-5490 Hydrosolv 4165 19 L» без границ колонок
        # неоднозначна и стабильно ломает вопросы про номера деталей.
        block_rect = fitz.Rect(b["bbox"])
        if kind in ("para", "step"):
            inside = None
            for idx, region in enumerate(table_regions):
                if region.intersects(block_rect) and block_rect.get_area():
                    overlap = (region & block_rect).get_area() / block_rect.get_area()
                    if overlap > 0.6:
                        inside = idx
                        break
            if inside is not None:
                if inside in emitted_tables:
                    continue
                emitted_tables.add(inside)
                rows = extract_table_rows(page, table_regions[inside])
                if rows:
                    elements.append(
                        Element("table", "", page_no, col, y_top, {"rows": rows})
                    )
                    continue
                kind = "table_row"  # сетка не разобралась — старое поведение

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
    profile = detect_heading_profile(doc)
    last = last_page or doc.page_count
    for pno in range(first_page - 1, last):
        yield from extract_page(doc[pno], pno + 1, profile)
    doc.close()
