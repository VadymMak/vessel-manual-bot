"""
Сборка элементов страницы в чанки-процедуры.

ГЛАВНОЕ ПРАВИЛО: чанк = одна целая процедура, а не фиксированные N токенов.

Почему так, а не «нарезать по 512 токенов»:
  * Процедура `Air Shutoff - Test` занимает страницы 68–70. Нарезка по
    страницам или по размеру разорвёт её на середине шага 4, и модель
    выдаст механику половину процедуры как полную.
  * Блок WARNING относится к конкретным шагам. Если он уедет в соседний
    чанк, ответ про очистку сердцевины охладителя придёт БЕЗ предупреждения
    о предельном давлении 205 kPa. Это документация на судовой двигатель —
    цена такой ошибки не «неточный ответ».

Отсюда инварианты, которые модуль обязан соблюдать:
  1. Граница чанка — только ID модуля Caterpillar (i########) или заголовок H1.
     Никогда не середина нумерованного списка.
  2. Процедура склеивается через границы страниц и колонок.
  3. Блоки WARNING/NOTICE всегда остаются внутри чанка своей процедуры.
  4. Если процедура длиннее лимита — режем по подзаголовкам H2 и ДУБЛИРУЕМ
     блоки безопасности в каждую часть. Дублирование текста дешевле
     потерянного предупреждения.
  5. Таблица не разрывается: её строки идут одним куском.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .extractor import Element

# Порог разбиения длинной процедуры. ~4 символа на токен.
MAX_CHUNK_CHARS = 8000

# Виды элементов, которые не могут начинать новый фрагмент при разбиении
_BODY_KINDS = {"para", "step", "table", "table_row", "illustration", "warning", "notice"}

RE_PART_NUMBER = re.compile(r"\b(\d{3}-\d{4}|\d[A-Z]-\d{4})\b")
RE_GRAPHIC_ID = re.compile(r"\b(g\d{8})\b")
RE_MODELS = re.compile(r"\b(3[59]\d{2}[BC])\b")
RE_ADEM = re.compile(r"\bADEM\s+(I{1,3}V?)\b")


@dataclass
class Chunk:
    """Единица индексации: одна процедура или её логическая часть."""
    heading: str
    icode: str | None
    section: str | None
    page_start: int
    page_end: int
    chunk_type: str                      # procedure | table | reference
    content: str                         # текст для эмбеддинга, в порядке чтения
    smcs_codes: list[str] = field(default_factory=list)
    part_numbers: list[str] = field(default_factory=list)
    illustrations: list[str] = field(default_factory=list)
    applicable_models: list[str] = field(default_factory=list)
    control_module: str | None = None
    safety_blocks: list[dict] = field(default_factory=list)
    step_count: int = 0
    part_index: int = 0                  # 0 = процедура целиком, 1..N = части
    part_total: int = 1

    @property
    def has_warning(self) -> bool:
        return any(b["type"] == "WARNING" for b in self.safety_blocks)

    @property
    def citation(self) -> str:
        if self.page_start == self.page_end:
            return f"стр. {self.page_start}"
        return f"стр. {self.page_start}–{self.page_end}"


# ─── Группировка элементов по процедурам ─────────────────────────────────────

def group_by_procedure(elements: list[Element]) -> list[list[Element]]:
    """Разбить поток элементов на группы «одна процедура».

    Граница — элемент icode. Если документ начинается без icode
    (титул, оглавление), первая группа собирается до первого icode.
    """
    groups: list[list[Element]] = []
    current: list[Element] = []

    for el in elements:
        if el.kind == "icode" and current:
            groups.append(current)
            current = [el]
        else:
            current.append(el)

    if current:
        groups.append(current)
    return groups


# ─── Рендеринг текста чанка ──────────────────────────────────────────────────

def _render_table(rows: list[list[str]]) -> str:
    """Отрисовать таблицу как Markdown с явными границами колонок.

    Плоская строка «1U-5490 Hydrosolv 4165 19 L (5 US gal)» неоднозначна:
    на вопрос «номер детали у Hydrosolv 4165» модель не может отличить номер
    от названия и от объёма. С разделителями и шапкой связь «столбец → значение»
    становится явной. На SEBU7844-37 без этого стабильно проваливались все вопросы
    по номерам деталей при корректно найденном чанке.

    Строки, объединённые на всю ширину (заголовок таблицы), выводим как подпись.
    """
    width = max(len(r) for r in rows)
    out: list[str] = []
    header_done = False

    for row in rows:
        if len(row) == 1 and width > 1:
            out.append(row[0])          # объединённая ячейка — название таблицы
            continue
        cells = list(row) + [""] * (width - len(row))
        out.append("| " + " | ".join(cells) + " |")
        if not header_done:
            out.append("|" + "|".join([" --- "] * width) + "|")
            header_done = True

    return "\n".join(out)


def _render(elements: list[Element]) -> str:
    """Собрать текст чанка, сохранив порядок чтения и разметив структуру."""
    lines: list[str] = []
    in_table = False

    for el in elements:
        if el.kind in ("icode", "smcs"):
            continue

        if el.kind != "table_row" and in_table:
            in_table = False

        if el.kind == "h1":
            lines.append(f"# {el.text}")
        elif el.kind == "h2":
            lines.append(f"## {el.text}")
        elif el.kind == "h3":
            lines.append(f"### {el.text}")
        elif el.kind == "warning":
            lines.append(f"WARNING: {el.text}")
        elif el.kind == "notice":
            body = el.text[len("NOTICE"):].strip()
            lines.append(f"NOTICE: {body}")
        elif el.kind == "table":
            rows = el.meta.get("rows")
            if rows:
                lines.append(_render_table(rows))
            elif el.text:
                lines.append(f"[{el.text}]")   # подпись «Table 38»
            in_table = True
        elif el.kind == "table_row":
            lines.append(f"  | {el.text}")
            in_table = True
        elif el.kind == "illustration":
            lines.append(f"[{el.text}]")
        else:
            lines.append(el.text)

    return "\n\n".join(lines)


def _collect_meta(elements: list[Element], chunk: Chunk) -> None:
    """Заполнить метаданные чанка из его элементов."""
    smcs: list[str] = []
    parts: set[str] = set()
    graphics: list[str] = []
    models: set[str] = set()

    for el in elements:
        if el.kind == "smcs":
            smcs.extend(el.meta.get("codes", []))
        if el.kind == "illustration" and "graphic_id" in el.meta:
            graphics.append(el.meta["graphic_id"])

        # Таблица хранит содержимое в meta["rows"], а не в .text — сканируем и его.
        # Иначе номера деталей из таблиц не попадут в part_numbers, а это поле
        # проиндексировано в GIN и используется для фильтрации по номеру.
        searchable = el.text
        if el.kind == "table" and el.meta.get("rows"):
            searchable += " " + " ".join(
                cell for row in el.meta["rows"] for cell in row
            )

        parts.update(RE_PART_NUMBER.findall(searchable))
        models.update(RE_MODELS.findall(searchable))

    # Модуль управления берём из подзаголовка H2 сегмента, а не из первого
    # попавшегося упоминания: заголовок процедуры может перечислять оба
    # варианта ("ADEM II or ADEM III"), и тогда метка вышла бы неверной.
    h2_text = " ".join(el.text for el in elements if el.kind == "h2")
    found = RE_ADEM.findall(h2_text) or RE_ADEM.findall(
        " ".join(el.text for el in elements if el.kind == "h1")
    )
    if len(set(found)) == 1:
        chunk.control_module = f"ADEM {found[0]}"

    chunk.smcs_codes = smcs
    chunk.part_numbers = sorted(parts)
    chunk.illustrations = graphics
    chunk.applicable_models = sorted(models)
    chunk.step_count = sum(1 for el in elements if el.kind == "step")


def _safety_blocks(elements: list[Element]) -> list[dict]:
    """Собрать блоки безопасности, склеив соседние строки одного блока."""
    blocks: list[dict] = []
    buffer: list[str] = []
    kind: str | None = None

    def flush() -> None:
        nonlocal buffer, kind
        if buffer and kind:
            blocks.append({"type": kind, "text": " ".join(buffer)})
        buffer, kind = [], None

    for el in elements:
        if el.kind == "warning":
            if kind != "WARNING":
                flush()
                kind = "WARNING"
            buffer.append(el.text)
        elif el.kind == "notice":
            flush()
            blocks.append({"type": "NOTICE", "text": el.text[len("NOTICE"):].strip()})
        else:
            flush()

    flush()
    return blocks


def _classify_chunk(elements: list[Element], section: str | None) -> str:
    if any(el.kind == "step" for el in elements):
        return "procedure"
    if any(el.kind in ("table", "table_row") for el in elements):
        return "table"
    return "reference"


# ─── Разбиение длинной процедуры ─────────────────────────────────────────────

def _has_steps(segment: list[Element]) -> bool:
    return any(el.kind == "step" for el in segment)


def _variant_level(elements: list[Element]) -> str | None:
    """Найти уровень подзаголовков, разделяющий ВАРИАНТЫ одной процедуры.

    Вариант — это самостоятельная последовательность шагов под своим
    подзаголовком. Примеры из SEBU7844-37:

      * `Air Shutoff - Test` → H2 «ADEM II» (5 шагов) и H2 «ADEM III» (8 шагов)
      * `Coolant - Change` → H2 «Flush» → H3 «Systems Filled with Cat ELC…»
        (шаги 1–5) и H3 «Systems Filled with Cat DEAC…» (шаги 3–9)

    Второй случай показывает, почему нельзя опираться на «нумерация начинается
    с единицы»: Caterpillar нумерует второй вариант с тройки. Надёжный признак —
    два и более подзаголовка одного уровня, под каждым из которых есть шаги.

    Возвращает 'h2' или 'h3' — уровень, по которому надо резать, либо None.
    """
    for level in ("h2", "h3"):
        positions = [i for i, el in enumerate(elements) if el.kind == level]
        if len(positions) < 2:
            continue
        bounds = positions + [len(elements)]
        with_steps = sum(
            _has_steps(elements[s:e]) for s, e in zip(bounds, bounds[1:])
        )
        if with_steps >= 2:
            return level
    return None


def _split_long(elements: list[Element], _depth: int = 0) -> list[list[Element]]:
    """Разрезать процедуру по подзаголовкам H2.

    Режем в ДВУХ случаях, и второй важнее первого:

    1. Размер превысил лимит.

    2. ВАРИАНТЫ ПРОЦЕДУРЫ. Если под H2 идут собственные последовательности
       шагов, начинающиеся с «1.», — это разные варианты одной процедуры
       (`Air Shutoff - Test` для ADEM II и для ADEM III: 5 шагов и 8 шагов
       с независимой нумерацией). Их обязательно разделять НЕЗАВИСИМО ОТ
       РАЗМЕРА: иначе на вопрос «как проверить air shutoff на ADEM III»
       вернётся чанк с обеими процедурами сразу, модель их смешает,
       а метка модуля управления будет от первого варианта — то есть неверной.

    Блоки безопасности всей процедуры дублируются в каждую часть — правило 4.
    Если H2 нет, а процедура длинная — не режем: лучше длинный чанк, чем
    список шагов, разорванный посередине.
    """
    too_long = sum(len(el.text) for el in elements) > MAX_CHUNK_CHARS
    level = _variant_level(elements)

    if not (too_long or level):
        return [elements]

    # Режем по уровню вариантов; если вариантов нет, но чанк длинный — по H2.
    split_level = level or "h2"
    h2_positions = [i for i, el in enumerate(elements) if el.kind == split_level]

    if not h2_positions:
        # Нечего резать по структуре. Для справочных блоков без шагов
        # допускаем разбиение по абзацам, для процедур — оставляем как есть.
        if any(el.kind == "step" for el in elements):
            return [elements]
        return _split_by_size(elements)

    header = elements[: h2_positions[0]]
    safety = [el for el in header if el.kind in ("warning", "notice")]
    # Сохраняем ВСЕ заголовки выше уровня разреза: при делении по H3 контекст
    # родительского H2 («Flush») обязан остаться, иначе вариант теряет смысл.
    title = [el for el in header if el.kind in ("h1", "h2", "h3", "smcs")]
    lead = [el for el in header if el.kind in ("para", "illustration", "table", "table_row")]

    parts: list[list[Element]] = []
    bounds = h2_positions + [len(elements)]
    for i, (start, end) in enumerate(zip(bounds, bounds[1:])):
        segment = elements[start:end]
        # Заголовки и предупреждения повторяются в каждой части;
        # вводный текст и иллюстрации — только в первой, чтобы не раздувать.
        prefix = title + safety + (lead if i == 0 else [])
        parts.append(prefix + segment)

    # Рекурсия: внутри части могут остаться варианты уровнем ниже
    # (H2 «Flush» → два H3 по типу охлаждающей жидкости).
    final: list[list[Element]] = []
    for part in parts:
        oversized = sum(len(el.text) for el in part) > MAX_CHUNK_CHARS
        if _depth < 2 and (_variant_level(part) or oversized):
            deeper = _split_long(part, _depth=_depth + 1)
            if len(deeper) > 1:
                final.extend(deeper)
                continue
        if oversized and not _has_steps(part):
            final.extend(_split_by_size(part))
        else:
            final.append(part)

    return final


def _split_by_size(elements: list[Element]) -> list[list[Element]]:
    """Запасное разбиение по границам абзацев — только для текста без шагов.

    Разрез разрешён НЕ в любом месте: серия строк таблицы обязана остаться
    целой (правило 5). Иначе справочная таблица из 27 строк — например,
    интервалы обслуживания — разъезжается на два чанка, и поиск вернёт
    половину, выглядящую как полная.
    """
    title = [el for el in elements if el.kind in ("h1", "smcs")]
    body = [el for el in elements if el.kind not in ("h1", "smcs")]

    parts: list[list[Element]] = []
    current: list[Element] = []
    size = 0

    for i, el in enumerate(body):
        prev_is_table = i > 0 and body[i - 1].kind in ("table", "table_row")
        this_is_table = el.kind in ("table", "table_row")
        # Внутри таблицы резать нельзя
        can_break = not (prev_is_table and this_is_table)

        if size + len(el.text) > MAX_CHUNK_CHARS and current and can_break:
            parts.append(title + current)
            current, size = [], 0

        current.append(el)
        size += len(el.text)

    if current:
        parts.append(title + current)
    return parts or [elements]


# ─── Основная сборка ─────────────────────────────────────────────────────────

def build_chunks(elements: list[Element], sections: dict[int, str] | None = None) -> list[Chunk]:
    """Собрать чанки из потока элементов документа."""
    sections = sections or {}
    chunks: list[Chunk] = []

    for group in group_by_procedure(elements):
        heading_el = next((el for el in group if el.kind == "h1"), None)
        if heading_el is None:
            continue  # титул, оглавление, колонтитулы — не индексируем

        icode_el = next((el for el in group if el.kind == "icode"), None)
        body = [el for el in group if el.kind in _BODY_KINDS or el.kind in ("h1", "h2", "h3", "smcs")]
        if not body:
            continue

        pages = [el.page for el in group]
        section = sections.get(min(pages))
        segments = _split_long(body)

        for idx, segment in enumerate(segments, start=1):
            seg_pages = [el.page for el in segment] or pages
            chunk = Chunk(
                heading=heading_el.text,
                icode=icode_el.text if icode_el else None,
                section=section,
                page_start=min(seg_pages),
                page_end=max(seg_pages),
                chunk_type=_classify_chunk(segment, section),
                content=_render(segment),
                safety_blocks=_safety_blocks(segment),
                part_index=idx if len(segments) > 1 else 0,
                part_total=len(segments),
            )
            _collect_meta(segment, chunk)
            chunks.append(chunk)

    return chunks
