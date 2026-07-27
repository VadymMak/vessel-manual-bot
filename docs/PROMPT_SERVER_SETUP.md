# Промпт для Claude в VS Code — развернуть проект на trading-server

**Важно:** этот промпт выполняется в окне VS Code, **подключённом к серверу через
Remote-SSH**, а не на Mac. Убедитесь, что в левом нижнем углу написано
`SSH: trading-server`.

Скопируйте всё, что ниже разделителя, в чат Claude-расширения.

---

Разворачиваем проект `vessel-manual-bot` на этом сервере. Разработка переезжает
сюда с macOS насовсем.

## Про машину

- Ubuntu, **Ryzen 9 6000-й серии** (8 ядер / 16 потоков), **64 ГБ RAM**
- Дискретной GPU нет. Есть интегрированная **Radeon 680M**, но ROCm официально
  не поддерживает `gfx1035` — **не пытайся использовать GPU**, всё на CPU
- База данных — **Neon в облаке**, схема уже создана, локальный PostgreSQL
  не нужен и ставить его не надо

## КРИТИЧНО: на этом сервере работает боевой торговый бот

Под PM2 крутятся процессы `researcher-paper` и, возможно, `researcher-live`,
они пишут в Neon реальные данные. **Ничего из этого не трогай:**

- не останавливай и не перезапускай процессы PM2
- не трогай `/home/bot/` и его venv
- **не ставь пакеты глобально** — только в venv нашего проекта
- перед тяжёлой индексацией ограничь число потоков, чтобы не задушить бота
  (см. ниже про `OMP_NUM_THREADS`)

Проверь `pm2 list` в начале и в конце — состояние процессов должно совпасть.

## Что делать

### 1. Клонировать репозиторий

```bash
git clone https://github.com/VadymMak/vessel-manual-bot.git
cd vessel-manual-bot
```

### 2. Убрать ограничения macOS из `requirements.txt`

Сейчас там пины `numpy<2` и `transformers>=4.40,<5`. Они существуют по одной
причине: на Intel-Mac PyTorch заморожен на 2.2.2, потому что сборки под
macOS x86_64 прекращены в январе 2024. **На Linux x86_64 этого ограничения нет** —
доступен torch 2.6+, и все три стены, которые мы прошли на Mac (numpy C API,
transformers 5.x с требованием torch ≥ 2.4, `torch.load` с требованием ≥ 2.6),
здесь просто не существуют.

Замени соответствующие строки на версии с маркерами окружения, чтобы один файл
работал на обеих платформах:

```
torch>=2.2,<2.3;       sys_platform == "darwin" and platform_machine == "x86_64"
torch>=2.6;            sys_platform != "darwin" or platform_machine != "x86_64"
numpy<2;               sys_platform == "darwin" and platform_machine == "x86_64"
numpy>=1.26;           sys_platform != "darwin" or platform_machine != "x86_64"
transformers>=4.40,<5; sys_platform == "darwin" and platform_machine == "x86_64"
transformers>=4.40;    sys_platform != "darwin" or platform_machine != "x86_64"
```

Обрати внимание: PEP 508 не поддерживает `not` в маркерах, поэтому отрицание
записано через `!=` с `or`. Сохрани существующие комментарии с объяснением
причины — они не устарели, просто теперь применяются только к Mac.

### 3. Окружение

Используй системный Python, если он ≥ 3.10 (проверь `python3 --version`),
иначе поставь `python3.12` из `deadsnakes`.

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

Проверь, что встало то, что ожидается:

```bash
./.venv/bin/python -c "import torch, numpy, transformers; \
  print('torch', torch.__version__, '| numpy', numpy.__version__, \
        '| transformers', transformers.__version__)"
```

Ожидается torch ≥ 2.6, numpy 2.x, transformers 5.x. Если FlagEmbedding при этом
не импортируется — вот тогда возвращай пины, но сначала проверь как есть.

### 4. Файл `.env`

`.env` в git не хранится. Создай из шаблона и попроси у меня два значения:

```bash
cp .env.example .env
```

Заполнить нужно `DB_DSN` (строка подключения Neon, я дам) и `OPENAI_API_KEY`.
Остальные параметры оставь как в шаблоне. Значения в кавычках — в DSN есть `&`.

### 5. Ограничить потребление CPU

Индексация загрузит все ядра и может помешать торговому боту. Добавь в `.env`:

```
OMP_NUM_THREADS=6
MKL_NUM_THREADS=6
```

Шесть из шестнадцати потоков — боту останется запас. Проверь, что `rag/config.py`
или `rag/embedder.py` их учитывает; если нет, выставь через `torch.set_num_threads()`
при загрузке модели и объясни мне, где именно поставил.

### 6. Проверки

```bash
make check-py     # должен найти .venv/bin/python
make check-env    # DB_DSN и OPENAI_API_KEY
make verify       # этап 1: ровно 5 строк OK, 161 чанк
./.venv/bin/python -c "from FlagEmbedding import BGEM3FlagModel; print('OK')"
```

`make verify` работает без сети и без базы — это хорошая первая проверка.

### 7. Замер скорости

Прежде чем запускать полный прогон, замерь одну операцию и покажи мне число:

```bash
time ./.venv/bin/python -c "
from rag.embedder import Embedder
import time
e = Embedder()
e.encode(['test'])                      # прогрев, загрузка модели
t = time.time(); e.encode(['x'*2000] * 8); print('8 чанков:', round(time.time()-t, 1), 'с')
"
```

На Mac было около 2.1 с на чанк. Хочу понять реальный выигрыш до того,
как запускать indexing и eval.

## Чего не делать

- **Не запускай `make load` и `make eval`** — я сделаю это сам после отчёта
- Не ставь PostgreSQL: база в Neon
- Не пытайся задействовать Radeon 680M через ROCm или HSA_OVERRIDE
- Не трогай процессы PM2, `/home/bot/`, их venv и конфиги
- Не меняй код в `ingestion/`, `rag/`, `eval/` — задача только про окружение

## Отчёт

Таблицей: версии Python и ключевых пакетов, результат каждой проверки,
время на замере скорости, состояние `pm2 list` до и после.
Если что-то не сошлось — покажи как есть, не замазывай.
