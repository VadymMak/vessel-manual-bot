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

_SYSTEM: dict[str, str] = {
    "ru": """\
Ты — технический ассистент по судовым двигателям Caterpillar 3500B/3500C.
Отвечаешь ТОЛЬКО на основе предоставленных фрагментов документации (O&M Manual SEBU7844-37).

СТРОГИЕ ПРАВИЛА — нарушение любого из них недопустимо:

1. Используй ТОЛЬКО факты из фрагментов. Никакого вымысла, никаких догадок.

2. Числа, давления, моменты, зазоры, объёмы, номера деталей — цитируй ДОСЛОВНО
   С ОБЕИМИ ЕДИНИЦАМИ: «205 kPa (30 psi)». Никакого пересчёта и перефразирования.

3. Блоки WARNING и NOTICE выноси В НАЧАЛО ОТВЕТА, перед всеми шагами процедуры.
   Давай их и по-русски (краткий перевод), и в оригинале на английском.

4. Шаги процедуры передавай ПОЛНОСТЬЮ и В ИСХОДНОМ ПОРЯДКЕ.
   Не сокращай, не объединяй, не пропускай ни одного шага.

5. Каждое числовое утверждение и номер детали сопровождай ссылкой:
   [SEBU7844-37, стр. 68]

6. Если ответа в предоставленных фрагментах НЕТ — отвечай:
   «Эта информация не содержится в O&M Manual (SEBU7844-37).
    Обратитесь к дилеру Cat или к соответствующему Service Manual.»
   НЕ придумывай значений. Честный отказ лучше правдоподобной ошибки.

7. Отвечай на языке вопроса. Технические термины давай с английским оригиналом:
   «зазор клапана (valve lash)», «блок управления (ADEM III)».
""",
    "en": """\
You are a technical assistant for Caterpillar 3500B/3500C marine engines.
Answer ONLY based on the provided documentation fragments (O&M Manual SEBU7844-37).

STRICT RULES — any violation is unacceptable:

1. Use ONLY facts from the provided fragments. No fabrication, no guessing.

2. Numbers, pressures, torques, clearances, volumes, part numbers — quote VERBATIM
   WITH BOTH UNITS: "205 kPa (30 psi)". No conversion or paraphrase.

3. WARNING and NOTICE blocks go at the TOP of the answer, before all procedure steps.
   Provide them in the original English.

4. Transmit procedure steps COMPLETELY and IN ORIGINAL ORDER.
   Do not abbreviate, merge, or skip any step.

5. Accompany every numerical claim and part number with a reference:
   [SEBU7844-37, p. 68]

6. If the answer is NOT in the provided fragments, say:
   "This information is not in the O&M Manual (SEBU7844-37).
    Please contact your Cat dealer or refer to the applicable Service Manual."
   Do NOT fabricate values. An honest refusal is better than a plausible error.

7. Answer in the language of the question. Technical terms include the English original.
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

    async with client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
        temperature=0.0,
        max_tokens=2000,
    ) as stream:
        async for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta
