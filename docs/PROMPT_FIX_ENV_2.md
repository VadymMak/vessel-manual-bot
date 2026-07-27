# Промпт для Claude в VS Code — починить окружение и довести до рабочего состояния

Скопируйте всё, что ниже разделителя, в чат Claude-расширения.

---

Твой аудит окружения был верным, диагноз обоих конфликтов подтверждён. План
согласован — выполняй. Но сначала прими один факт, которого у тебя не было,
он меняет трактовку.

## Недостающий факт

`torch 2.2.2` стоит не по случайности и не из-за неудачного `pip install`.
**Это последняя версия PyTorch с колёсами под macOS x86_64.** Сборки под
Intel-макинтоши объявили устаревшими в январе 2024 и после 2.2.2 не выпускают.
Машина Intel, значит torch здесь заморожен на 2.2.2 навсегда, и «обновить torch
до 2.4, чтобы устроить transformers 5.x» — невозможный путь.

Следствие: `numpy<2` и `transformers<5` — не выбор из соображений
стабильности, а **вынужденное следствие потолка torch на этой машине**.
Это важно записать в комментариях, иначе через год кто-то снимет пины
как устаревший рудимент и всё сломается заново.

Второе следствие: **эти ограничения — свойство ноутбука, а не проекта.**
На Ubuntu-сервере (x86_64 Linux) доступен torch 2.4+ и ни один пин не нужен.
Пины пока делаем общими, без environment markers — `FlagEmbedding 1.4.0`
импортирует `transformers.trainer` даже для инференса, и совместимость
с пятой веткой transformers нигде не проверена. Один согласованный набор версий
на обеих платформах даёт на один источник расхождений меньше.

## Задача

### 1. `requirements.txt`

Добавь `numpy<2` с комментарием, объясняющим происхождение ограничения:

```
# numpy 1.x ОБЯЗАТЕЛЕН, и причина не в numpy.
# torch 2.2.2 собран против NumPy 1.x C API (_ARRAY_API). На numpy 2.x импорт
# torch выдаёт «Failed to initialize NumPy: _ARRAY_API not found» и torch
# остаётся полуживым.
# Обновить torch нельзя: 2.2.2 — последняя версия с колёсами под macOS x86_64,
# сборки под Intel-Mac прекращены в январе 2024.
# На Linux-сервере ограничение неактуально — там доступен torch 2.4+.
numpy<2
```

Пин `transformers>=4.40,<5` уже есть — проверь, что комментарий рядом с ним
тоже объясняет причину (transformers 5.x требует torch >= 2.4).

### 2. Создать `.venv`

Базовый интерпретатор — Homebrew python@3.11 (3.11.14), тот, что ты определил
как единственный с полным набором пакетов:

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

Ставь **только** в `.venv`. Ничего не доустанавливай в Homebrew
site-packages, conda, pyenv и системные Python — там сейчас сломанное
состояние, и трогать его не нужно, оно просто перестанет использоваться.

### 3. Поправить автопоиск интерпретатора в `Makefile`

Сейчас в переменной `PYTHON` есть список кандидатов, но **`.venv/bin/python`
в нём нет**. Добавь его ПЕРВЫМ кандидатом, до всех остальных — иначе `make`
продолжит находить сломанный Homebrew-питон вместо чистого venv.

### 4. Закрепить в редакторе и в репозитории

- `.vscode/settings.json`: `"python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"`
- `.gitignore`: убедись, что `.venv/` там есть
- `README.md`: раздел «Окружение» — какой интерпретатор, как активировать,
  почему пины, и что на Linux-сервере они не нужны

### 5. Проверки

Выполни по порядку и покажи вывод каждой:

```bash
./.venv/bin/python --version
./.venv/bin/python -c "import numpy, torch, transformers; \
  print('numpy', numpy.__version__); \
  print('torch', torch.__version__); \
  print('transformers', transformers.__version__)"
```

Ожидается: numpy 1.x, torch 2.2.2, transformers 4.x.
Предупреждения про `_ARRAY_API not found` быть НЕ должно.

```bash
./.venv/bin/python -c "from FlagEmbedding import BGEM3FlagModel; print('BGEM3FlagModel OK')"
./.venv/bin/python -c "from FlagEmbedding import FlagReranker; print('FlagReranker OK')"
```

Оба импорта обязаны пройти без трейсбека и без строки
«Disabling PyTorch because PyTorch >= 2.4 is required».

```bash
make check-py     # должен показать путь .venv/bin/python
make check-env    # DB_DSN и OPENAI_API_KEY
make verify       # этап 1 не сломался: ровно 5 строк OK
```

### 6. Отчёт

Короткой таблицей: что изменено в каких файлах, итоговые версии пакетов,
результат каждой проверки. Если что-то не сошлось — не замазывай, покажи как есть.

## Чего не делать

- **Не запускай `make load`.** Он потянет bge-m3 на 2.3 ГБ — я сделаю это сам
  после того, как увижу отчёт.
- Не чини Homebrew site-packages, conda и системные Python — они больше
  не используются.
- Не добавляй environment markers по платформам в `requirements.txt`.
- Не трогай код в `ingestion/`, `rag/`, `eval/` — задача только про окружение
  и сборку.
- Не меняй `.env` — он заполнен и работает.
