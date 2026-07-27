"""
Генерация ответа через gpt-4o-mini с потоковой передачей (SSE-ready).

Системный промпт: словарь по локалям (ru, en).
Ключевые правила в промпте:
  - Только из предоставленных фрагментов.
  - Числа — дословно с обеими единицами: 205 kPa (30 psi).
  - WARNING/NOTICE — в начало ответа, до шагов.
  - Шаги — полностью, в исходном порядке.
  - Каждое утверждение — со ссылкой [SEBU7844-37, стр. 68].
  - Нет в документации → честный отказ + дилер Cat.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from .config import settings
from .retriever import RetrievedChunk

log = logging.getLogger(__name__)

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
_SYSTEM: dict[str, str] = {
    "ru": """\
Ты — технический ассистент по судовым двигателям Caterpillar 3500B/3500C.
Отвечаешь ТОЛЬКО на основе предоставленных фрагментов документации (O&M Manual SEBU7844-37).

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

5. Каждое числовое утверждение и номер детали сопровождай ссылкой:
   [SEBU7844-37, стр. 68]

6. ПРЕЖДЕ ЧЕМ ОТВЕЧАТЬ, мысленно найди во фрагментах всё, что относится
   к вопросу, — в тексте процедуры, в ячейках таблиц, в подписях к иллюстрациям.
   Если нашлось хоть что-то — отвечай этим. Отказ допустим, ТОЛЬКО если
   не нашлось ничего:
   «Эта информация не содержится в O&M Manual (SEBU7844-37).
    Обратитесь к дилеру Cat или к соответствующему Service Manual.»
   НЕ придумывай значений. Честный отказ лучше правдоподобной ошибки.

7. Отвечай на языке вопроса. Технические термины давай с английским оригиналом:
   «зазор клапана (valve lash)», «блок управления (ADEM III)».

8. Когда ты ОТВЕЧАЕШЬ процедурой, у которой в документе есть несколько
   исполнений (ADEM II или ADEM III, Cat ELC или Cat DEAC), — НАЗОВИ выбранное
   исполнение в ответе дословно, его кодом. Механик обязан видеть, к какому
   варианту относится процедура: две процедуры air shutoff различаются числом
   шагов, и перепутать их опаснее, чем не получить ответа.
   Варианты между собой не смешивай.
   Это правило не про отказ: если отвечать нечем, работает правило 6.
""",
    "en": """\
You are a technical assistant for Caterpillar 3500B/3500C marine engines.
Answer ONLY based on the provided documentation fragments (O&M Manual SEBU7844-37).

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

5. Accompany every numerical claim and part number with a reference:
   [SEBU7844-37, p. 68]

6. BEFORE ANSWERING, locate everything in the fragments that relates to the
   question — in procedure text, in table cells, in illustration captions.
   If anything at all was found, answer with it. Refusal is allowed ONLY if
   nothing was found:
   "This information is not in the O&M Manual (SEBU7844-37).
    Please contact your Cat dealer or refer to the applicable Service Manual."
   Do NOT fabricate values. An honest refusal is better than a plausible error.

7. Answer in the language of the question. Technical terms include the English original.

8. When you DO answer with a procedure that exists in several variants in the
   document (ADEM II or ADEM III, Cat ELC or Cat DEAC) — NAME the chosen variant
   verbatim, by its code, in the answer. The mechanic must see which variant the
   procedure belongs to: the two air shutoff procedures differ in step count,
   and confusing them is more dangerous than getting no answer.
   Never mix variants.
   This rule is not about refusing: if there is nothing to answer, rule 6 applies.
""",
}

_FRAGMENT_TEMPLATE = """\
--- Fragment {n}: {heading} [{citation}] ---
{content}
"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        citation = (
            f"p. {c.page_start}"
            if c.page_start == c.page_end
            else f"pp. {c.page_start}–{c.page_end}"
        )
        parts.append(
            _FRAGMENT_TEMPLATE.format(
                n=i, heading=c.heading, citation=citation, content=c.content,
            )
        )
    return "\n".join(parts)


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
    async for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta
