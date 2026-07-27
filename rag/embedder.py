"""
BAAI/bge-m3: dense (1024) + sparse (lexical) embeddings, CPU-only, ленивая загрузка.

Бэкенд: FlagEmbedding с PyTorch float32.
  — fp16 на x86 CPU не ускоряет (нет hardware поддержки), используем float32.
  — ROCm для Radeon 680M (gfx1035) официально не поддерживается → только CPU.

ONNX int8 upgrade-path (если PyTorch окажется медленным):
  1. pip install optimum[onnxruntime]
  2. optimum-cli export onnx --model BAAI/bge-m3 --quantize int8 ./bge-m3-int8/
     Но: sparse-голова (MLM head) требует кастомного ONNX-постпроцессинга —
     нет готового решения в optimum 1.x. Пока PyTorch float32 на Ryzen 9 6000
     даёт ~200-400 мс на запрос, что приемлемо при 3-5 пользователях.
  3. Интерфейс EmbedResult/Embedder менять не придётся.

Cross-lingual: bge-m3 кросс-языковая — русский запрос ищет по английскому тексту.
Перевода запроса в пайплайне нет намеренно (правило из CLAUDE.md): перевод
искажает технические термины («зазор клапана» → «valve clearance», а в мануале
«valve lash» — разные термины в разных процедурах).

Формат ключей lexical_weights (FlagEmbedding _process_token_weights):
  for w, idx in zip(token_weights, input_ids):
      if idx not in unused_tokens and w > 0:
          idx = str(idx)   ← str(token_id), а не токен-строка
          result[idx] = max(result[idx], w)
  Ключи — "6", "12345" и т.д. Преобразование: int(key) + 1 (pgvector 1-based).

pgvector SPARSEVEC индексация 1-based ("like SQL arrays"):
  token_id ∈ [0, 250001] → pgvec_index = token_id + 1 ∈ [1, 250002]
  Применяется одинаково при индексации и при запросе → инвариант соблюдается.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

log = logging.getLogger(__name__)

# Размерность разрежённого вектора = размер словаря XLM-RoBERTa (backbone bge-m3)
VOCAB_SIZE: int = 250_002

# Лимиты длины: самый длинный чанк ~9247 симв. ≈ 2400 токенов.
# Паддинг батча по длинному элементу + квадратичный attention → лимит имеет значение.
MAX_LEN_INDEX: int = 2048   # при индексации чанков
MAX_LEN_QUERY: int = 512    # вопрос механика всегда короткий

_load_lock = threading.Lock()


@dataclass(slots=True)
class EmbedResult:
    dense: np.ndarray          # shape (1024,), dtype float32, L2-нормирован
    sparse: dict[int, float]   # pgvec_index (1-based) → weight (только ненулевые)


class Embedder:
    """
    Singleton-обёртка над BGEM3FlagModel.

    Модель грузится при первом вызове encode() — не при импорте.
    Это позволяет импортировать модуль без задержки (важно для CLI).
    Thread-safe: двойная проверка с _load_lock.
    """

    _instance: ClassVar[Embedder | None] = None
    _model: object | None

    def __new__(cls) -> Embedder:
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._model = None
            cls._instance = obj
        return cls._instance

    # ─── Загрузка модели ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        with _load_lock:
            if self._model is not None:  # двойная проверка после захвата лока
                return

            from .config import settings
            from FlagEmbedding import BGEM3FlagModel

            # Ограничение числа потоков. На сервере рядом крутится боевой
            # торговый бот (systemd: trading-bot, mexc-*), и индексация,
            # забравшая все 16 потоков, отбирает у него CPU.
            # OMP_NUM_THREADS из .env одного не хватает: переменная действует
            # только если выставлена ДО инициализации OpenMP-рантайма, а торч
            # к этому моменту уже импортирован (numpy/torch тянутся сверху).
            # torch.set_num_threads() работает в рантайме и потому надёжнее.
            # Настройка глобальна для процесса — реранкер (rag/reranker.py)
            # наследует её, если эмбеддер загрузился первым.
            import os
            import torch

            n_threads = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
            if n_threads > 0:
                torch.set_num_threads(n_threads)
                log.info("torch: ограничил интра-оп потоки до %d", n_threads)

            log.info("Загружаю bge-m3 на CPU (~30 с при первом запуске)…")
            try:
                # FlagEmbedding ≥ 1.3: поддерживает параметр devices
                self._model = BGEM3FlagModel(
                    settings.embed_model,
                    use_fp16=False,
                    devices=["cpu"],
                )
            except TypeError:
                # FlagEmbedding 1.2.x: нет параметра devices; CUDA отсутствует → CPU
                self._model = BGEM3FlagModel(
                    settings.embed_model,
                    use_fp16=False,
                )
            log.info("bge-m3 загружена.")

    # ─── Публичный API ────────────────────────────────────────────────────────

    def encode(
        self,
        texts: list[str],
        batch_size: int = 8,
        max_length: int = MAX_LEN_INDEX,
    ) -> list[EmbedResult]:
        """
        Закодировать список текстов за один проход модели.

        Возвращает dense (1024-мерный, L2-нормированный) и sparse
        (lexical weights, pgvector 1-based индексы).

        max_length: используй MAX_LEN_INDEX при индексации чанков,
                    MAX_LEN_QUERY при кодировании пользовательского запроса.
        """
        if not texts:
            return []
        self._load()

        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs: np.ndarray = output["dense_vecs"]           # (n, 1024)
        lex_weights: list[dict] = output["lexical_weights"]     # list[dict]

        return [
            EmbedResult(
                dense=dense.astype(np.float32),
                sparse=_to_pgvec_indices(lw),
            )
            for dense, lw in zip(dense_vecs, lex_weights)
        ]

    def encode_one(self, text: str, max_length: int = MAX_LEN_QUERY) -> EmbedResult:
        """Удобный метод для одного текста (запрос пользователя)."""
        return self.encode([text], batch_size=1, max_length=max_length)[0]


# ─── Функции для работы со sparse-вектором ───────────────────────────────────
# Вынесены из класса: не зависят от состояния модели и нужны в retriever/loader.

def _to_pgvec_indices(lexical_weights: dict) -> dict[int, float]:
    """
    Конвертировать выход FlagEmbedding в {pgvector_index: weight}.

    FlagEmbedding (_process_token_weights) отдаёт {str(token_id): weight}.
    Ключи — строковые представления token_id: "6", "12345" и т.п.
    Преобразование: int(key) + 1 — переход к 1-based индексации pgvector.
    Диапазон token_id 0..250001 → pgvec_index 1..250002 = SPARSEVEC(250002).
    """
    result: dict[int, float] = {}
    for key, weight in lexical_weights.items():
        w = float(weight)
        if w <= 0:
            continue
        try:
            tid = int(key)
        except (TypeError, ValueError):
            continue
        if not 0 <= tid < VOCAB_SIZE:
            continue
        idx = tid + 1  # 1-based
        if result.get(idx, 0.0) < w:
            result[idx] = w
    return result


def sparse_to_pgvector_literal(sparse: dict[int, float]) -> str:
    """
    Сериализовать разрежённый вектор в текстовый формат pgvector SPARSEVEC.

    Формат: '{1:0.5,2:0.3,...}/250002'
    Используется при вставке через psycopg3 с явным ::sparsevec cast.

    Пустой словарь → пустой вектор '{}':
      SELECT '{}/250002'::sparsevec;  -- допустим в pgvector
    Не вставляем фиктивные нули — pgvector хранит только ненулевые элементы.
    """
    if not sparse:
        return f"{{}}/{VOCAB_SIZE}"
    pairs = ",".join(f"{k}:{v:.8g}" for k, v in sorted(sparse.items()))
    return f"{{{pairs}}}/{VOCAB_SIZE}"
