#!/usr/bin/env bash
# Выгрузка bge-reranker-v2-m3 в ONNX + динамическая int8-квантизация.
#
# Результат (models/) не версионируется — 2.8 ГБ, воспроизводится этим скриптом
# за пару минут. Запускать после клона репозитория, если RERANKER_BACKEND=onnx.
#
# Только реранкер. Эмбеддер bge-m3 сюда добавлять не нужно: у него sparse-голова
# с кастомным постпроцессингом, готового экспорта в optimum нет, а в общем
# времени запроса эмбеддинг занимает меньше процента.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-./.venv/bin/python}
FP32=models/bge-reranker-onnx
INT8=models/bge-reranker-onnx-int8

$PYTHON -m pip install -q "optimum[onnxruntime]"

echo "→ Экспорт в ONNX (float32, ~2.2 ГБ)…"
$PYTHON -m optimum.commands.optimum_cli export onnx \
    --model BAAI/bge-reranker-v2-m3 \
    --task text-classification \
    "$FP32"

echo "→ Динамическая квантизация в int8 (~565 МБ)…"
$PYTHON - <<PY
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

src, dst = Path("$FP32"), Path("$INT8")
dst.mkdir(parents=True, exist_ok=True)
quantize_dynamic(
    model_input=src / "model.onnx",
    model_output=dst / "model.onnx",
    weight_type=QuantType.QInt8,
    extra_options={"EnableSubgraph": True},
)
PY

# Токенизатор квантизация не трогает, но рядом с сессией он нужен:
# rag/reranker.py грузит AutoTokenizer из той же папки, что и model.onnx.
for f in config.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json sentencepiece.bpe.model; do
    cp "$FP32/$f" "$INT8/$f"
done

echo "→ Готово. Включить: RERANKER_BACKEND=onnx в .env"
echo "   Откат на эталон: RERANKER_BACKEND=torch"
