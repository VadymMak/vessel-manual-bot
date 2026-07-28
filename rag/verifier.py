"""
Grounding-проверка без LLM (правило 8 из CLAUDE.md).

Каждое число с единицами и номер детали из ответа обязаны присутствовать
в извлечённом контексте. Не подтвердилось → блокируем выдачу.

Возвращает VerificationResult с перечнем неподтверждённых значений
для логирования и подсчёта метрик.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retriever import RetrievedChunk

# Числа с единицами измерения — то, что механик может использовать на практике
RE_MEASUREMENT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*"
    r"(?:kPa|psi|bar|°C|°F|N·?m|lb·?ft|lb·?in|kW|hp|rpm|"
    r"L|mL|US gal|US qt|gal|qt|mm|cm|m\b|in(?:ch)?)\b",
    re.IGNORECASE,
)

# Номера деталей Caterpillar: 174-6854, 1U-5490, 8T-0890
RE_PART = re.compile(r"\b\d{3}-\d{4}\b|\b\d[A-Z]-\d{4}\b")


@dataclass
class VerificationResult:
    ok: bool
    unverified: list[str] = field(default_factory=list)
    # Сколько значений вообще нашлось в ответе. Без этого числа ok=True
    # неотличимо от «регулярки не сматчили ничего»: и то, и другое даёт
    # пустой unverified, но первое означает проверку, а второе — её отсутствие.
    claims: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.ok:
            return "✓ grounding OK"
        return f"✗ unverified ({len(self.unverified)}): {', '.join(self.unverified)}"


def verify(answer: str, chunks: list[RetrievedChunk]) -> VerificationResult:
    """
    Извлечь из ответа все числовые утверждения и номера деталей.
    Проверить, что каждое встречается в объединённом контексте чанков.

    Нормализация при сравнении: удаляем пробелы внутри измерений,
    чтобы «205 kPa» и «205kPa» считались одним значением.
    """
    context_raw = "\n\n".join(c.content for c in chunks)
    context_norm = re.sub(r"\s+", "", context_raw)

    claims: list[str] = []
    claims.extend(m.group(0) for m in RE_MEASUREMENT.finditer(answer))
    claims.extend(m.group(0) for m in RE_PART.finditer(answer))

    unverified: list[str] = []
    for claim in claims:
        claim_norm = re.sub(r"\s+", "", claim)
        if claim_norm not in context_norm:
            unverified.append(claim)

    return VerificationResult(
        ok=len(unverified) == 0, unverified=unverified, claims=claims,
    )
