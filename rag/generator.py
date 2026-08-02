"""
Генерация ответа через gpt-4o-mini с потоковой передачей (SSE-ready).

Системный промпт: словарь по локалям (ru, en).
Ключевые правила в промпте:
  - Только из предоставленных фрагментов.
  - Числа — дословно с обеими единицами: 205 kPa (30 psi).
  - WARNING/NOTICE — в начало ответа, до шагов.
  - Шаги — полностью, в исходном порядке.
  - Каждое утверждение — со ссылкой вида [<публикация>, стр. 68], где
    публикация берётся из заголовка фрагмента, а не из промпта.
  - Нет в документации → честный отказ + дилер Cat.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from .config import settings
from .retriever import RetrievedChunk

log = logging.getLogger(__name__)

# Формулировка отказа из правила 6. Держим оба языка. Живёт здесь, а не в
# scorelog.py, потому что здесь же она и задаётся моделью в промпте: если
# правило 6 однажды перепишут, детектор обязан поехать в том же файле.
# scorelog.py импортирует отсюда.
_REFUSAL_MARKERS = (
    "не содержится в O&M Manual",
    "is not in the O&M Manual",
)


def looks_like_refusal(answer: str) -> bool:
    return any(m in answer for m in _REFUSAL_MARKERS)

# ПОРЯДОК ВНУТРИ ПРАВИЛА 6 — РЕЗУЛЬТАТ ИЗМЕРЕНИЯ, НЕ СТИЛИСТИКА.
# Прежняя формулировка («если ответа во фрагментах НЕТ — откажись») заставляла
# gpt-4o-mini отказывать на вопросы к ячейкам таблиц, даже когда нужный чанк
# приходил первым фрагментом: 0/3 на gs002/gs003/gs006 при 1, 2, 4 и 6
# фрагментах. Та же модель с тем же контекстом без правила 6 — 3/3.
# Переформулировка «отказывай, только если значения нет ни в тексте, ни в
# ячейке» НЕ помогла (0/3). Помог порядок «сначала найди всё по вопросу,
# потом решай, отказывать ли»: 3/3 на табличных вопросах при сохранении
# честных отказов на gs027–gs030.
# И ещё: сюда идёт РОВНО то, что читает модель. Пояснения к правкам держи
# в комментариях модуля — абзац, вписанный в текст промпта, модель прочтёт
# как инструкцию и поведение изменится.
#
# ПРАВИЛА 8 ЗДЕСЬ БОЛЬШЕ НЕТ, И ВОЗВРАЩАТЬ ЕГО НЕ НАДО. Оно требовало назвать
# исполнение процедуры (ADEM II / ADEM III, Cat ELC / Cat DEAC) в ответе.
# Теперь исполнение печатается механической шапкой из метаданных чанка
# (applicability_line ниже), и просить модель помнить об этом незачем.
#
# Промежуточная редакция «варианты не смешивай, отвечай шагами ровно одного —
# того, о котором спрашивают» выглядела безобидной страховкой, но замер
# 2026-07-28 (5 прогонов на вопрос, gpt-4o-mini, temperature=0) показал:
#     правило 8 сокращённое  gs031: 1/5 прошло, 4 отказа из 5
#     правило 8 удалено      gs031: 4/5 прошло, 1 отказ из 5
# gs031 («нужно ли выводить на высокие обороты при заполнении системы
# охлаждения») исполнения не называет. Модель, обязанная отвечать шагами
# «ровно одного варианта», не находила, какого именно, — и отказывала.
# Контрольные вопросы на обеих редакциях 5/5: gs007/gs008 (выбор исполнения),
# gs002/gs006 (таблицы), gs027/gs030 (честные отказы). То есть страховка
# ничего не защищала, а ложные отказы создавала.
_SYSTEM: dict[str, str] = {
    "ru": """\
Ты — технический ассистент по судовым двигателям и генераторным установкам Caterpillar.
Отвечаешь ТОЛЬКО на основе предоставленных фрагментов документации.
Фрагменты бывают ИЗ РАЗНЫХ ПУБЛИКАЦИЙ. Публикация каждого фрагмента названа
в его заголовке после слова SOURCE — это единственный источник ссылки.

СТРОГИЕ ПРАВИЛА — нарушение любого из них недопустимо:

1. Используй ТОЛЬКО факты из фрагментов. Никакого вымысла, никаких догадок.

2. Числа, давления, моменты, зазоры, объёмы, номера деталей — цитируй ДОСЛОВНО
   С ОБЕИМИ ЕДИНИЦАМИ: «205 kPa (30 psi)». Никакого пересчёта и перефразирования.

   Данные в таблицах — такой же факт документа, как обычный текст.
   Значение в ячейке таблицы является прямой цитатой из документа
   и может использоваться в ответе наравне с текстом процедуры.

3. Блоки WARNING и NOTICE выноси В НАЧАЛО ОТВЕТА, перед всеми шагами процедуры.
   Давай их и по-русски (краткий перевод), и в оригинале на английском.

4. Шаги процедуры передавай ПОЛНОСТЬЮ и В ИСХОДНОМ ПОРЯДКЕ.
   Не сокращай, не объединяй, не пропускай ни одного шага.

5. Каждое числовое утверждение и номер детали сопровождай ссылкой.
   Идентификатор публикации и страницу БЕРИ ИЗ ЗАГОЛОВКА ТОГО ФРАГМЕНТА,
   откуда взят факт. Заголовок выглядит так:
       --- Fragment 2 — SOURCE [<идентификатор>, p. <страница>] — <заголовок> ---
   значит ссылка на этот факт: [<идентификатор>, стр. <страница>].
   НЕ подставляй идентификатор по памяти и НЕ переноси его с другого фрагмента.

   Если ответ собран из фрагментов РАЗНЫХ публикаций — ссылайся на каждую
   отдельно, рядом с её фактом. НИКОГДА не объединяй страницы разных
   публикаций в один диапазон: диапазон «стр. 104-118», собранный из двух
   разных мануалов, — грубая ошибка, а не сокращение.

6. ПРЕЖДЕ ЧЕМ ОТВЕЧАТЬ, мысленно найди во фрагментах всё, что относится
   к вопросу, — в тексте процедуры, в ячейках таблиц, в подписях к иллюстрациям.
   Если нашлось хоть что-то — отвечай этим. Отказ допустим, ТОЛЬКО если
   не нашлось ничего:
   «Эта информация не содержится в O&M Manual.
    Обратитесь к дилеру Cat или к соответствующему Service Manual.»
   НЕ придумывай значений. Честный отказ лучше правдоподобной ошибки.

7. Отвечай на языке вопроса. Технические термины давай с английским оригиналом:
   «зазор клапана (valve lash)», «блок управления (ADEM III)».
""",
    "en": """\
You are a technical assistant for Caterpillar marine engines and generator sets.
Answer ONLY based on the provided documentation fragments.
Fragments may come FROM DIFFERENT PUBLICATIONS. The publication of each fragment
is named in its header after the word SOURCE — that is the only source of a reference.

STRICT RULES — any violation is unacceptable:

1. Use ONLY facts from the provided fragments. No fabrication, no guessing.

2. Numbers, pressures, torques, clearances, volumes, part numbers — quote VERBATIM
   WITH BOTH UNITS: "205 kPa (30 psi)". No conversion or paraphrase.

   Data in tables is as much a fact of the document as ordinary text.
   A value in a table cell is a direct quotation from the document
   and may be used in the answer on equal terms with procedure text.

3. WARNING and NOTICE blocks go at the TOP of the answer, before all procedure steps.
   Provide them in the original English.

4. Transmit procedure steps COMPLETELY and IN ORIGINAL ORDER.
   Do not abbreviate, merge, or skip any step.

5. Accompany every numerical claim and part number with a reference.
   TAKE the publication id and the page FROM THE HEADER OF THE FRAGMENT the fact
   came from. The header looks like this:
       --- Fragment 2 — SOURCE [<id>, p. <page>] — <heading> ---
   so the reference for that fact is: [<id>, p. <page>].
   Do NOT supply the id from memory and do NOT carry it over from another fragment.

   If the answer is assembled from fragments of DIFFERENT publications, reference
   each one separately, next to its own fact. NEVER merge pages of different
   publications into one range: a range "pp. 104-118" assembled from two
   different manuals is a gross error, not an abbreviation.

6. BEFORE ANSWERING, locate everything in the fragments that relates to the
   question — in procedure text, in table cells, in illustration captions.
   If anything at all was found, answer with it. Refusal is allowed ONLY if
   nothing was found:
   "This information is not in the O&M Manual.
    Please contact your Cat dealer or refer to the applicable Service Manual."
   Do NOT fabricate values. An honest refusal is better than a plausible error.

7. Answer in the language of the question. Technical terms include the English original.
""",
}

_FRAGMENT_TEMPLATE = """\
--- Fragment {n} — SOURCE [{doc}, {citation}] — {heading} ---
{content}
"""


def _publication_id(chunk: RetrievedChunk) -> str:
    """
    Идентификатор публикации для ссылки. ЕДИНСТВЕННОЕ МЕСТО, где он берётся.

    Сейчас — имя файла без расширения. Для SEBU7844-37.pdf это и есть код
    публикации, для C18-MarineGenSet.pdf — нет (настоящий код SEBU8118-06
    стоит на титульной странице, но в БД поля publication_no пока нет).
    Решение отложить поле — сознательное; когда оно появится, меняется
    ровно эта функция, а формат фрагмента и промпт остаются как есть.

    Пустое имя не подставляем: лучше явное «unknown source», по которому
    видно, что источник потерян, чем правдоподобный чужой код.
    """
    name = (chunk.doc_filename or "").strip()
    if not name:
        return "unknown source"
    return name[:-4] if name.lower().endswith(".pdf") else name


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """
    Контекст для модели. КАЖДЫЙ фрагмент несёт СВОЙ идентификатор публикации.

    Раньше во фрагмент шли только страницы. При двух мануалах в базе это
    давало неверные ссылки: модель не могла отличить фрагмент C18 от
    фрагмента SEBU7844-37 и подставляла код публикации из системного
    промпта, где он был константой. Замеренный результат — ответ, где шаги
    из C18-MarineGenSet (стр. 117–118) подписаны «[SEBU7844-37, pp. 104-118]»,
    хотя стр. 118 в SEBU7844-37 совсем о другом. Это нарушение правила 9
    и опаснее отказа: отказ виден, а неверная ссылка выглядит правильным
    ответом.
    """
    parts = []
    for i, c in enumerate(chunks, start=1):
        citation = (
            f"p. {c.page_start}"
            if c.page_start == c.page_end
            else f"pp. {c.page_start}–{c.page_end}"
        )
        parts.append(
            _FRAGMENT_TEMPLATE.format(
                n=i, doc=_publication_id(c), heading=c.heading,
                citation=citation, content=c.content,
            )
        )
    return "\n".join(parts)


# ─── Применимость ────────────────────────────────────────────────────────────
# Собирается МЕХАНИЧЕСКИ из метаданных чанка, а не просьбой к модели.
#
# Прежняя редакция правила 8 («назови исполнение дословно») стабилизировала
# gs008, но роняла табличные gs002/gs003/gs006, а расширенная формулировка
# обваливала honest_refusal с 5/5 до 1/5: модель принимала номер модели
# двигателя из вопроса (3512C) за исполнение. Размен исчезает целиком, если
# перестать просить модель это помнить: control_module и applicable_models
# лежат в БД у каждого чанка, это факт, а не рассуждение.
#
# Берём метаданные ПЕРВОГО чанка после реранкинга: трассировка показывает, что
# ответ строится на нём (для gs008 это id=230, ADEM II, логит +5.79 против
# +5.03 у соседнего ADEM III).
_APPLICABILITY_LABEL = {"ru": "Применимо к", "en": "Applies to"}
_CM_PREFIX = {"ru": "двигатели с ", "en": "engines with "}

# Сколько символов ответа подождать, прежде чем решать, печатать ли шапку.
# Печатать её нужно ПЕРЕД ответом, а знать про отказ можно только увидев текст,
# поэтому начало потока придерживаем. Отказ по правилу 6 — это весь ответ
# целиком, его маркер стоит в первом предложении (~55 символов), так что 300
# с запасом хватает, а задержка неощутима: это первые токены, они приходят
# быстрее всего.
_REFUSAL_PROBE_CHARS = 300


def applicability_line(
    chunks: list[RetrievedChunk],
    lang: str = "ru",
) -> str | None:
    """
    Строка применимости из метаданных первого чанка.

    Применимость по умолчанию задаёт ДОКУМЕНТ, метаданные чанка её сужают:

        есть control_module        → «двигатели с ADEM II» (+ модели чанка)
        иначе есть applicable_models → перечислить их
        иначе                       → applicable_models документа (титул)

    Из 164 чанков SEBU7844-37 собственную разметку несут 13, и это не пробел:
    большинство процедур применимо ко всему семейству и потому ничего
    не перечисляет — процедуре очистки охладителя незачем называть шесть
    моделей. Поэтому третий уровень и нужен: без него строка была бы
    осмысленна на 13 чанках вместо 161.

    None остаётся только для документа, у которого и на титуле ничего нет:
    пустое «Применимо к: —» хуже, чем ничего, а выдумывать применимость
    к судовому двигателю нельзя.
    """
    if not chunks:
        return None
    c = chunks[0]
    label = _APPLICABILITY_LABEL.get(lang, _APPLICABILITY_LABEL["ru"])
    prefix = _CM_PREFIX.get(lang, _CM_PREFIX["ru"])

    def _clean(values: list[str] | None) -> list[str]:
        return [v.strip() for v in (values or []) if v and v.strip()]

    parts: list[str] = []
    cm = (c.control_module or "").strip()
    own_models = _clean(c.applicable_models)

    if cm:
        parts.append(f"{prefix}{cm}")
        # Модели чанка рядом с исполнением: у air shutoff ADEM II это {3500B},
        # и «двигатели с ADEM II · 3500B» точнее, чем одно из двух.
        if own_models:
            parts.append(", ".join(own_models))
    elif own_models:
        parts.append(", ".join(own_models))
    else:
        doc_models = _clean(c.doc_models)
        if doc_models:
            parts.append(", ".join(doc_models))

    if not parts:
        return None
    return f"{label}: " + " · ".join(parts)


def applicability_for(
    query: str,
    chunks: list[RetrievedChunk],
    answer: str,
) -> str | None:
    """
    Строка применимости для УЖЕ ГОТОВОГО ответа — структурное поле для UI
    этапа 4 и для скор-лога.

    Отдельно от applicability_line, потому что учитывает отказ: «Применимо к:
    ADEM II. Этой информации нет в мануале» — абсурд, и он испортил бы
    честные отказы. Ровно то же условие применяет generate() к тексту,
    так что структурное поле и текст ответа не разойдутся.
    """
    if looks_like_refusal(answer):
        return None
    return applicability_line(chunks, _detect_lang(query))


def control_modules_in(chunks: list[RetrievedChunk]) -> list[str]:
    """Различные исполнения среди чанков, по алфавиту. Для скор-лога."""
    return sorted({
        (c.control_module or "").strip()
        for c in chunks
        if (c.control_module or "").strip()
    })


def _detect_lang(query: str) -> str:
    """Простая эвристика: >10% кириллицы → ru, иначе en."""
    if not query:
        return "ru"
    cyrillic = sum(1 for ch in query if "Ѐ" <= ch <= "ӿ")
    return "ru" if cyrillic / len(query) > 0.1 else "en"


async def generate(
    query: str,
    chunks: list[RetrievedChunk],
    lang: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Сгенерировать ответ потоком токенов (async generator).

    Пример использования:
        async for token in generate(query, chunks):
            print(token, end="", flush=True)
    """
    lang = lang or _detect_lang(query)
    system = _SYSTEM.get(lang, _SYSTEM["ru"])
    context = _build_context(chunks)

    if lang == "ru":
        user_msg = f"Фрагменты документации:\n\n{context}\n\nВопрос: {query}"
    else:
        user_msg = f"Documentation fragments:\n\n{context}\n\nQuestion: {query}"

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    log.info("Генерирую ответ (lang=%s, фрагментов=%d)…", lang, len(chunks))

    stream = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
        temperature=0.0,
        max_tokens=2000,
    )
    # Шапка применимости идёт ПЕРЕД ответом, но печатать её нельзя, пока не
    # ясно, не отказ ли это. Поэтому придерживаем начало потока (см. коммент
    # к _REFUSAL_PROBE_CHARS), решаем один раз и дальше стримим как обычно.
    line = applicability_line(chunks, lang)
    head = ""
    decided = False

    async for event in stream:
        delta = event.choices[0].delta.content
        if not delta:
            continue
        if decided:
            yield delta
            continue
        head += delta
        if len(head) < _REFUSAL_PROBE_CHARS:
            continue
        if line and not looks_like_refusal(head):
            yield f"{line}\n\n"
        decided = True
        yield head
        head = ""

    # Ответ оказался короче окна — решение принимаем на том, что пришло.
    # Сюда попадают как раз отказы: они короткие.
    if not decided:
        if line and not looks_like_refusal(head):
            yield f"{line}\n\n"
        if head:
            yield head
