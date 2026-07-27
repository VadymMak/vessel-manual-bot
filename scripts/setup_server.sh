#!/usr/bin/env bash
# Подготовка trading-server: PostgreSQL 16 + pgvector + база проекта.
# Запускать НА СЕРВЕРЕ:  bash scripts/setup_server.sh
#
# Скрипт идемпотентен: повторный запуск ничего не ломает.

set -euo pipefail

DB_NAME="${DB_NAME:-vessel}"
DB_USER="${DB_USER:-vessel}"

echo "═══ 1. Проверка, что уже установлено ═══"
if command -v psql >/dev/null 2>&1; then
    echo "PostgreSQL уже стоит: $(psql --version)"
else
    echo "PostgreSQL не найден, ставлю 16 из репозитория PGDG…"
    sudo apt-get update -qq
    sudo apt-get install -y curl ca-certificates gnupg lsb-release
    sudo install -d /usr/share/postgresql-common/pgdg
    sudo curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
        https://www.postgresql.org/media/keys/ACCC4CF8.asc
    echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] \
https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
        | sudo tee /etc/apt/sources.list.d/pgdg.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y postgresql-16 postgresql-client-16
fi

PG_MAJOR="$(psql --version | grep -oE '[0-9]+' | head -1)"
echo "Мажорная версия: ${PG_MAJOR}"

echo
echo "═══ 2. Расширение pgvector ═══"
if ! sudo -u postgres psql -tAc \
     "SELECT 1 FROM pg_available_extensions WHERE name='vector'" | grep -q 1; then
    echo "Ставлю postgresql-${PG_MAJOR}-pgvector…"
    sudo apt-get install -y "postgresql-${PG_MAJOR}-pgvector"
else
    echo "pgvector уже доступен."
fi

echo
echo "═══ 3. Пароль ═══"
# Пароль генерируем здесь. Никакого «взять откуда-то» — его просто не существует,
# пока вы его не создадите.
if [ -z "${DB_PASSWORD:-}" ]; then
    DB_PASSWORD="$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)"
    echo "Сгенерирован новый пароль."
else
    echo "Использую пароль из переменной DB_PASSWORD."
fi

echo
echo "═══ 4. Пользователь и база ═══"
# Отдельные роль и база: не смешиваем с базой торгового бота.
sudo -u postgres psql <<SQL
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${DB_USER}') THEN
        CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASSWORD}';
    ELSE
        ALTER ROLE ${DB_USER} PASSWORD '${DB_PASSWORD}';
    END IF;
END
\$\$;
SQL

if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
    sudo -u postgres createdb -O "${DB_USER}" "${DB_NAME}"
    echo "База ${DB_NAME} создана."
else
    echo "База ${DB_NAME} уже существует."
fi

# CREATE EXTENSION требует прав суперпользователя — делаем от postgres.
sudo -u postgres psql -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo
echo "═══ 5. Проверка ═══"
sudo -u postgres psql -d "${DB_NAME}" -tAc \
    "SELECT 'pgvector ' || extversion FROM pg_extension WHERE extname='vector';"
sudo -u postgres psql -d "${DB_NAME}" -tAc \
    "SELECT '{1:0.5,7:0.2}/250002'::sparsevec IS NOT NULL AS sparsevec_ok;"

echo
echo "═══ ГОТОВО ═══"
echo
echo "Впишите в .env НА СЕРВЕРЕ:"
echo
echo "DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@localhost:5432/${DB_NAME}"
echo
echo "Пароль показан один раз — сохраните его сейчас."
echo
echo "Дальше:  psql \"\$DATABASE_URL\" -f migrations/001_init.sql"
