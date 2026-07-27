"""
Реранкинг топ-20 кандидатов → топ-6 с помощью BAAI/bge-reranker-v2-m3 (CPU, float32).

Ожидаемое время на Ryzen 9 6000 при 20 кандидатах: 0.5–1.5 с.
Если >2 с — уменьши rrf_top_k в config.py с 20 до 10.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import ClassVar

from .retriever import RetrievedChunk

log = logging.getLogger(__name__)

_load_lock = threading.Lock()


class Reranker:
    """Singleton-обёртка над FlagReranker. Ленивая загрузка, thread-safe."""

    _instance: ClassVar[Reranker | None] = None
    _model: object | None

    def __new__(cls) -> Reranker:
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._model = None
            cls._instance = obj
        return cls._instance

    def _load(self) -> None:
        if self._model is not None:
            return
        with _load_lock:
            if self._model is not None:
                return
            from .config import settings
            from FlagEmbedding import FlagReranker

            log.info("Загружаю bge-reranker-v2-m3 на CPU…")
            try:
                self._model = FlagReranker(
                    settings.reranker_model,
                    use_fp16=False,
                    devices=["cpu"],
                )
            except TypeError:
                self._model = FlagReranker(
                    settings.reranker_model,
                    use_fp16=False,
                )
            log.info("Реранкер загружен.")

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        """
        Отранжировать чанки по релевантности к запросу.

        Возвращает top_n чанков в порядке убывания score.
        """
        if not chunks:
            return []
        self._load()

        pairs = [[query, c.content] for c in chunks]

        t0 = time.monotonic()
        scores: list[float] = self._model.compute_score(pairs, normalize=True)
        elapsed = time.monotonic() - t0

        log.info(
            "Реранкер: %d кандидатов → топ %d за %.2f с",
            len(chunks), top_n, elapsed,
        )
        if elapsed > 2.0:
            log.warning(
                "Реранкер слишком медленный (%.2f с). "
                "Уменьши rrf_top_k в config.py с %d до 10.",
                elapsed, len(chunks),
            )

        ranked = sorted(zip(scores, chunks), key=lambda x: -x[0])
        result = [chunk for _, chunk in ranked[:top_n]]

        for i, (score, chunk) in enumerate(ranked[:top_n]):
            log.debug(
                "  #%d score=%.4f  %s  (p.%d–%d)",
                i + 1, score, chunk.heading, chunk.page_start, chunk.page_end,
            )

        return result
