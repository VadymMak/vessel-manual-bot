"""
Реранкинг топ-20 кандидатов → топ-6 с помощью BAAI/bge-reranker-v2-m3 (CPU).

Два бэкенда, переключаются через RERANKER_BACKEND в .env:

  torch — эталон. FlagReranker, float32. ~36 с на 20 кандидатов.
  onnx  — ONNX Runtime, динамическая int8-квантизация. ~8 с на тех же данных.

torch остаётся рабочим и остаётся дефолтом в .env.example: на нём сверяются
скоры и на него откатываются, если int8 когда-нибудь начнёт врать. Замер
2026-07-28 на 12 вопросах: топ-1 совпал 12/12, медиана расхождения логитов
0.20, максимум 0.68; расхождения живут в хвосте (логиты −6…−10), где порядок
всё равно не значим. Golden set: 31/32 на обоих, падает один и тот же вопрос
по одной и той же причине.

Из-за сдвига логитов rag/scorelog.py пишет, каким бэкендом посчитан скор:
калибровать порог отказа по перемешанным шкалам нельзя.

──────────────────────────────────────────────────────────────────────────────
ЗАМЕРЫ 2026-07-28, Ryzen 9 6900HX, 8 потоков, 20 кандидатов. Все четыре факта
ниже противоречат интуиции, и все четыре проверены. Если решите «оптимизировать»
что-то из этого обратно — сначала повторите замер, а не рассуждение.

1. batch_size=1 БЫСТРЕЕ ВСЕГО. Не «приемлемо», а именно быстрее:
       bs=1 → 8.03 с   bs=4 → 9.67 с   bs=8 → 13.44 с   bs=20 → 15.12 с
   На GPU батч экономит запуски ядер; на CPU он упирается в кэш — батч из
   20 пар по 512 токенов туда не помещается, и время растёт вдвое. Плюс при
   bs=1 нет паддинга: каждая пара считается ровно на своей длине.
   Заодно исчезает разброс: bs=1 держится в 7.93–8.08 с, bs=10 гуляет
   от 10.6 до 16.9 с.

2. FlagEmbedding СЧИТАЕТ ДВАЖДЫ. В compute_score_single_gpu блок «adjust batch
   size» делает полный прямой проход по первому батчу ради ловли OOM — и
   ВЫБРАСЫВАЕТ результат, после чего основной цикл считает то же заново.
   При дефолтном batch_size=128 и 20 кандидатах первый батч — это все пары.
   Голый прямой проход стоил 17.6 с против 35.7 с у compute_score. На GPU
   с подбором батча под память это осмысленно, на CPU — 17 секунд в никуда.
   ONNX-путь идёт мимо обёртки, отсюда ускорение 4.5x, а не 2x, которые
   обещает сама квантизация.

3. max_length УЖЕ БЫЛ 512 — это дефолт FlagReranker, и 18 чанков из 20 при нём
   и так обрезаются (19 633 токена → 9 560). Хвосты длинных чанков реранкер
   не видел никогда, «отключить обработку хвостов» нечего. Токенизация стоит
   0.011 с против 35 с счёта: искать выигрыш в ней бессмысленно.

4. БЕНЧМАРКИ БЭКЕНДОВ — ТОЛЬКО В ОТДЕЛЬНЫХ ПРОЦЕССАХ. Если в одном процессе
   загрузить оба, torch-граф на 568M float32 остаётся резидентным и портит
   цифры ONNX: 14 с вместо 8 с. Первый прогон сравнения показал ускорение
   2.5x вместо настоящих 4.5x именно из-за этого.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import ClassVar

from .retriever import RetrievedChunk

log = logging.getLogger(__name__)

_load_lock = threading.Lock()


class Reranker:
    """Singleton-обёртка над реранкером. Ленивая загрузка, thread-safe."""

    _instance: ClassVar[Reranker | None] = None
    _model: object | None
    _session: object | None
    _tokenizer: object | None
    backend: str

    def __new__(cls) -> Reranker:
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._model = None
            obj._session = None
            obj._tokenizer = None
            obj.backend = "torch"
            cls._instance = obj
        return cls._instance

    # ─── загрузка ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        from .config import settings

        if settings.reranker_backend == "onnx":
            self._load_onnx()
        else:
            self._load_torch()

    def _load_torch(self) -> None:
        # backend переставляем ДО проверки на «уже загружено»: иначе после
        # onnx→torch в одном процессе поле осталось бы равным "onnx",
        # и scorelog приписал бы torch-скоры чужой шкале.
        self.backend = "torch"
        if self._model is not None:
            return
        with _load_lock:
            if self._model is not None:
                return
            from .config import settings
            from FlagEmbedding import FlagReranker

            log.info("Загружаю bge-reranker-v2-m3 на CPU (torch, float32)…")
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
            self.backend = "torch"
            log.info("Реранкер загружен (torch).")

    def _load_onnx(self) -> None:
        self.backend = "onnx"
        if self._session is not None:
            return
        with _load_lock:
            if self._session is not None:
                return
            import os
            from pathlib import Path

            import onnxruntime as ort
            from transformers import AutoTokenizer

            from .config import settings

            path = Path(settings.reranker_onnx_path)
            if not (path / "model.onnx").is_file():
                # Молчаливый откат на torch запрещён: он в четыре раза медленнее
                # и пишет скоры в другой шкале. Лучше упасть здесь.
                raise FileNotFoundError(
                    f"RERANKER_BACKEND=onnx, но {path / 'model.onnx'} не найден. "
                    "Выгрузи модель (см. scripts/export_reranker_onnx.sh) "
                    "или поставь RERANKER_BACKEND=torch."
                )

            log.info("Загружаю bge-reranker-v2-m3 на CPU (onnx, int8)…")
            opts = ort.SessionOptions()
            # То же ограничение, что OMP_NUM_THREADS в Makefile: вторая половина
            # машины занята торговым ботом. ort своё окружение не наследует.
            opts.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                str(path / "model.onnx"),
                opts,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = AutoTokenizer.from_pretrained(str(path))
            self.backend = "onnx"
            log.info("Реранкер загружен (onnx int8).")

    # ─── счёт ────────────────────────────────────────────────────────────────

    def _logits_onnx(self, pairs: list[list[str]]) -> list[float]:
        """Сырые логиты в порядке pairs. Пары сортируются по длине — меньше паддинга."""
        import numpy as np

        from .config import settings

        tok = self._tokenizer
        max_len = settings.reranker_max_length
        bs = settings.reranker_batch_size

        # truncation="only_second": режем ЧАНК, а не запрос. Обратный порядок
        # обрезал бы вопрос пользователя и сделал сравнение бессмысленным.
        enc = tok(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            truncation="only_second",
            max_length=max_len,
            padding=False,
        )
        lengths = [len(x) for x in enc["input_ids"]]
        order = sorted(range(len(pairs)), key=lambda i: -lengths[i])

        logits: list[float] = [0.0] * len(pairs)
        for start in range(0, len(order), bs):
            idx = order[start:start + bs]
            batch = tok.pad(
                {
                    "input_ids": [enc["input_ids"][i] for i in idx],
                    "attention_mask": [enc["attention_mask"][i] for i in idx],
                },
                padding=True,
                return_tensors="np",
            )
            out = self._session.run(
                None,
                {
                    "input_ids": batch["input_ids"].astype(np.int64),
                    "attention_mask": batch["attention_mask"].astype(np.int64),
                },
            )[0].reshape(-1)
            for pos, i in enumerate(idx):
                logits[i] = float(out[pos])

        return logits

    # ─── публичный интерфейс ─────────────────────────────────────────────────

    def logits(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[float], float]:
        """
        Сырые логиты в порядке chunks и время счёта — для rag/trace.py.

        Без сигмоиды: она монотонна, ранга не меняет, а трассировке удобнее
        шкала, в которой мерялось разделение «есть ответ / нет ответа».
        """
        if not chunks:
            return [], 0.0
        self._load()
        pairs = [[query, c.content] for c in chunks]

        t0 = time.monotonic()
        if self.backend == "onnx":
            scores = self._logits_onnx(pairs)
        else:
            scores = self._model.compute_score(pairs, normalize=False)
            if isinstance(scores, float):
                scores = [scores]
        return [float(s) for s in scores], time.monotonic() - t0

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
        return self.rerank_with_scores(query, chunks, top_n)[0]

    def rerank_with_scores(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> tuple[list[RetrievedChunk], list[float]]:
        """
        То же, что rerank, но отдаёт и сырые скоры — для rag/scorelog.py.

        Отдельный метод, а не изменение сигнатуры rerank: у rerank есть
        вызывающие, которым скоры не нужны, и ломать их незачем.
        """
        if not chunks:
            return [], []
        self._load()

        pairs = [[query, c.content] for c in chunks]

        t0 = time.monotonic()
        if self.backend == "onnx":
            # normalize=True у FlagReranker — это ровно сигмоида. Держим ту же
            # шкалу: scorelog.py восстанавливает логит обратным преобразованием.
            scores: list[float] = [
                1.0 / (1.0 + math.exp(-x)) for x in self._logits_onnx(pairs)
            ]
        else:
            scores = self._model.compute_score(pairs, normalize=True)
        elapsed = time.monotonic() - t0

        log.info(
            "Реранкер (%s): %d кандидатов → топ %d за %.2f с",
            self.backend, len(chunks), top_n, elapsed,
        )
        # Порог по бэкендам разный: torch на 20 кандидатах честно стоит ~35 с,
        # предупреждать об этом на каждом запросе — шум. Для onnx 15 с уже
        # аномалия и повод смотреть, не подгрузился ли float32-граф.
        limit = 45.0 if self.backend == "torch" else 12.0
        if elapsed > limit:
            log.warning(
                "Реранкер медленнее ожидаемого (%.2f с при %d кандидатах, "
                "бэкенд %s). Проверь RERANKER_BACKEND и RRF_TOP_K.",
                elapsed, len(chunks), self.backend,
            )

        ranked = sorted(zip(scores, chunks), key=lambda x: -x[0])
        result = [chunk for _, chunk in ranked[:top_n]]
        top_scores = [score for score, _ in ranked[:top_n]]

        for i, (score, chunk) in enumerate(ranked[:top_n]):
            log.debug(
                "  #%d score=%.4f  %s  (p.%d–%d)",
                i + 1, score, chunk.heading, chunk.page_start, chunk.page_end,
            )

        return result, top_scores
