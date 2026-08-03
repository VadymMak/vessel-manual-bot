PDF ?= docs/SEBU7844-37.pdf

# На macOS обычно соседствуют несколько Python (framework с python.org,
# Homebrew, conda), и «python3» нередко указывает на самый старый из них.
# Поэтому ищем интерпретатор, который одновременно >= 3.10 И видит зависимости
# проекта. Переопределить: make load PYTHON=/путь/к/python
PYTHON ?= $(shell \
  for p in .venv/bin/python \
           python3.13 python3.12 python3.11 python3.10 python3 python \
           /usr/local/opt/python@3.13/bin/python3.13 \
           /usr/local/opt/python@3.12/bin/python3.12 \
           /usr/local/opt/python@3.11/bin/python3.11 \
           /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 \
           /opt/homebrew/bin/python3.11; do \
    command -v $$p >/dev/null 2>&1 || continue; \
    $$p -c 'import sys,click,psycopg' >/dev/null 2>&1 \
      && $$p -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null \
      && { echo $$p; break; }; \
  done)

# Если ничего с зависимостями не нашлось — берём просто свежий Python,
# чтобы check-py выдал осмысленную подсказку про pip install.
ifeq ($(strip $(PYTHON)),)
PYTHON := $(shell \
  for p in python3.13 python3.12 python3.11 python3.10 python3; do \
    command -v $$p >/dev/null 2>&1 && \
      $$p -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null \
      && { echo $$p; break; }; \
  done)
endif
ifeq ($(strip $(PYTHON)),)
PYTHON := python3
endif

# .env читается только внутри рецептов, через subshell.
# Значения в .env ДОЛЖНЫ быть в кавычках: DSN содержит & и ?, иначе bash
# споткнётся при source. python-dotenv кавычки снимает сам, так что
# Python и shell читают один и тот же файл одинаково.
LOAD_ENV = set -a; . ./.env; set +a;

# Ограничение потоков BLAS/OpenMP. Это переменные окружения ОС, а не конфиг
# приложения, поэтому им место здесь, а не в .env: rag/config.py работает
# с extra="forbid", и лишний ключ в .env уронит загрузку настроек.
# Ровно 8: по замерам на этой машине 8 потоков быстрее и 6 (2.29 против
# 2.46 с/чанк), и 16 (2.60 — SMT только мешает). Второй половины машины
# торговому боту (systemd: trading-bot, mexc-*) хватает с запасом.
# rag/embedder.py читает OMP_NUM_THREADS из окружения процесса и дублирует
# ограничение через torch.set_num_threads().
#
# ПОРЯДОК ЦЕЛЕЙ В СТРОКЕ ЗНАЧИМ: `load` первым словом строки GNU make 4.x
# разбирает как директиву load (загрузка динамического объекта) и падает с
# «load: cannot open shared object file». Поэтому load стоит не первым.
query load eval eval-retrieval: export OMP_NUM_THREADS=8
query load eval eval-retrieval: export MKL_NUM_THREADS=8

.PHONY: check-py install ingest verify migrate load query eval eval-retrieval families psql dev clean check-env

# ─── Этап 1: ingestion ───────────────────────────────────────────────────────
install: 
	$(PYTHON) -m pip install -r requirements.txt

ingest:
	$(PYTHON) -m ingestion.cli $(if $(F),$(F),$(PDF))

verify:
	$(PYTHON) verify_chunks.py $(if $(F),$(F),$(PDF))

# ─── Этап 2: RAG-ядро ────────────────────────────────────────────────────────

# Проверка, что .env заполнен и совпадает с rag/config.py
check-py:
	@$(PYTHON) -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null \
	  || { echo ""; \
	       echo "Нужен Python >= 3.10, найден: $$($(PYTHON) --version 2>&1)  ($(PYTHON))"; \
	       echo "В коде есть @dataclass(slots=True) — это 3.10+."; \
	       echo "Укажите интерпретатор явно:  make load PYTHON=/путь/к/python3.12"; \
	       echo ""; exit 1; }
	@$(PYTHON) -c "import click, psycopg, pydantic_settings" 2>/dev/null \
	  || { echo ""; \
	       echo "Python подходит ($$($(PYTHON) --version 2>&1)), но зависимости не установлены."; \
	       echo "Поставьте их именно в ЭТОТ интерпретатор:"; \
	       echo ""; \
	       echo "  $(PYTHON) -m pip install -r requirements.txt"; \
	       echo ""; exit 1; }
	@echo "OK: $$($(PYTHON) --version 2>&1)  ($(PYTHON))"

check-env: check-py
	@test -f .env || { echo "Нет .env — сделайте: cp .env.example .env"; exit 1; }
	@$(LOAD_ENV) test -n "$$DB_DSN" || { echo "DB_DSN пуст. Имя переменной должно быть DB_DSN (не DATABASE_URL)"; exit 1; }
	@$(LOAD_ENV) test -n "$$OPENAI_API_KEY" || { echo "OPENAI_API_KEY пуст"; exit 1; }
	@$(LOAD_ENV) echo "OK: DB_DSN и OPENAI_API_KEY заданы"

# Применить схему. Только ПРЯМОЙ эндпоинт Neon, без -pooler:
# DDL и CREATE EXTENSION требуют полноценной сессии.
migrate: check-env
	@$(LOAD_ENV) psql "$$DB_DSN" -v ON_ERROR_STOP=1 -f migrations/001_init.sql
	@$(LOAD_ENV) psql "$$DB_DSN" -c "SELECT '{}/250002'::sparsevec IS NOT NULL AS empty_sparsevec_ok;"

# Произвольный запрос: make psql Q="SELECT count(*) FROM chunks"
psql: check-env
	@$(LOAD_ENV) psql "$$DB_DSN" $(if $(Q),-c "$(Q)",)

load: check-env
	$(PYTHON) -m rag.cli --load

# make query Q="как очистить aftercooler"
# make query Q="air shutoff" CM="ADEM III"
query: check-env
	$(PYTHON) -m rag.cli "$(Q)" $(if $(CM),--cm "$(CM)",) $(if $(M),--models "$(M)",)

# make eval                 — весь golden set, без фильтра
# make eval M="3512B"       — с фильтром по модели двигателя (SQL WHERE)
# make eval CAT=part_number — только одна категория
eval: check-env
	$(PYTHON) -m eval.run $(if $(CAT),--category $(CAT),) $(if $(M),--models "$(M)",)

# ─── Ретривальная оценка ─────────────────────────────────────────────────────
# Детерминированная, БЕЗ OpenAI: эмбеддинг запроса, Neon, реранкер. Одного
# прогона достаточно, результат точный. Мерить правки поиска надо этим, а не
# make eval: генерация дрейфует у провайдера и топит сигнал поиска.
#
# make eval-retrieval               — все размеченные вопросы
# make eval-retrieval M="3512B"     — с фильтром по модели двигателя
# make eval-retrieval CAT=correct_variant
# make eval-retrieval IDS=gs007,gs008
eval-retrieval: check-env
	$(PYTHON) -m eval.retrieval $(if $(CAT),--category $(CAT),) \
		$(if $(M),--models "$(M)",) $(if $(IDS),--ids $(IDS),)

# ─── Таблица семейств моделей ────────────────────────────────────────────────
# Выводится из корпуса, константой нигде не записана. Смотреть после КАЖДОЙ
# индексации нового мануала и сверять с его титульной страницей: строка
# «3500B → 3508B, 3512B, 3516B» обязана совпадать с тем, что напечатано
# над серийными префиксами. Ненулевой код возврата — семейство, не покрывающее
# ни одной модели своего документа, то есть недостижимые чанки.
#
# make families              — вся таблица
# make families M="3512B"    — что раскрытие даёт этому запросу
families: check-env
	$(PYTHON) -m scripts.family_table $(if $(M),--models "$(M)",)

dev:
	uvicorn rag.api:app --reload --host 0.0.0.0 --port 8001

clean:
	rm -f chunks.json .prefix_cache.json
	find . -name __pycache__ -type d -exec rm -rf {} +
