"""
JSONL-лог скоров реранкера — сырьё для будущей калибровки порога отказа.

ПОРОГОВОЙ ОТСЕЧКИ ЗДЕСЬ НЕТ И ПОКА БЫТЬ НЕ ДОЛЖНО. На замере 2026-07-27
(12 вопросов) разделение выглядит чистым: медиана логита top-1 равна −2.88
там, где ответа в мануале нет, и +3.28 там, где есть, пересечений нет.
Но отрицательных примеров всего четыре, а худший положительный (gs006) лежит
на +0.15, почти на нуле. Порог по такой выборке — это подгонка под шум.
Поэтому здесь только сбор. Через месяц эксплуатации данных хватит на честную
калибровку, и тогда решение об отсечке будет опираться на распределение,
а не на четыре точки.

Формат — одна JSON-строка на запрос:
  ts, source, backend, question, chunk_ids, top1_sigmoid, sigmoids,
  top1_logit, logits, refused, answer_chars

Порядок значений совпадает с порядком chunk_ids (то есть после реранкинга).

Про backend. С 2026-07-28 реранкер умеет считать на ONNX int8 (rag/reranker.py),
и квантизация сдвигает логиты: на замере 12 вопросов медиана расхождения 0.32,
максимум 0.76. Для ранжирования это неважно — порядок топ-1 сохраняется, —
но для калибровки порога отказа важно очень: 0.3 логита это заметная доля
расстояния до худшего положительного примера (gs006, +0.15). Строки с разными
backend нельзя складывать в одну выборку, иначе откалиброванный порог окажется
средним по двум шкалам и не подойдёт ни к одной. Фильтруйте по этому полю.

Про две шкалы. Боевой реранкер вызывает compute_score(normalize=True) и отдаёт
СИГМОИДЫ, а разделение, ради которого всё затевалось, измерялось в ЛОГИТАХ.
Менять боевой вызов ради логирования нельзя, поэтому логиты восстанавливаются
обратным преобразованием ln(p/(1−p)). Оно точное, но у насыщенных значений
(p → 0 или 1) разрешение float теряется, поэтому в файл пишутся обе шкалы:
сигмоида — как есть, логит — как удобная для калибровки производная.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import settings
from .retriever import RetrievedChunk

log = logging.getLogger(__name__)

# Дозапись из нескольких потоков: строки короткие, но рвать их нельзя.
_write_lock = threading.Lock()

# Формулировка отказа из правила 6 системного промпта. Держим оба языка:
# по ней и только по ней определяется refused — эвристики вроде «ответ короткий»
# врут (короткий ответ на вопрос о номере детали — это норма).
_REFUSAL_MARKERS = (
    "не содержится в O&M Manual",
    "is not in the O&M Manual",
)


def looks_like_refusal(answer: str) -> bool:
    return any(m in answer for m in _REFUSAL_MARKERS)


def _active_backend() -> str:
    """
    Каким бэкендом реально посчитаны скоры.

    Спрашиваем сам реранкер, а не settings.reranker_backend: настройка говорит,
    чего просили, а атрибут — что загрузилось. Разойтись они не должны
    (_load_onnx падает, если модели нет), но лог обязан фиксировать факт,
    а не намерение — иначе однажды перепутанная выборка обойдётся дороже.
    Импорт локальный: reranker тянет за собой torch, а лог зовут и оттуда.
    """
    from .reranker import Reranker

    return Reranker().backend


def _to_logit(p: float) -> float:
    """Обратная сигмоида с зажимом: p ровно 0 или 1 дало бы ±inf и сломало JSON."""
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def log_query(
    question: str,
    chunks: list[RetrievedChunk],
    scores: list[float],
    answer: str,
    source: str = "cli",
) -> None:
    """
    Записать один запрос в JSONL.

    scores — сигмоиды из compute_score(normalize=True), в порядке chunks.
    Ошибка записи не должна ронять ответ пользователю: лог диагностический,
    а не транзакционный.
    """
    try:
        logits = [_to_logit(s) for s in scores]
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "backend": _active_backend(),
            "question": question,
            "chunk_ids": [c.id for c in chunks],
            "top1_sigmoid": round(scores[0], 6) if scores else None,
            "sigmoids": [round(s, 6) for s in scores],
            "top1_logit": round(logits[0], 4) if logits else None,
            "logits": [round(x, 4) for x in logits],
            "refused": looks_like_refusal(answer),
            "answer_chars": len(answer),
        }
        line = json.dumps(record, ensure_ascii=False)
        path = Path(settings.score_log_path)
        with _write_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 — лог не имеет права ломать запрос
        log.warning("Не смог записать скор-лог: %s", exc)
