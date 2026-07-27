from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="forbid" ЗАДАН ЯВНО И НАМЕРЕННО, это не значение по умолчанию.
    # Опечатка в имени переменной (классика: DATABASE_URL вместо DB_DSN) при
    # ignore проглатывается молча, pydantic берёт дефолт localhost:5432, и вы
    # ищете проблему в сети вместо .env. Один такой случай уже стоил часа.
    # Следствие: в .env не должно быть ничего, кроме полей этого класса.
    # Переменные окружения ОС (OMP_NUM_THREADS, MKL_NUM_THREADS) живут
    # в Makefile, а не здесь.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="forbid"
    )

    # PostgreSQL
    db_dsn: str = "postgresql://vessel:vessel@localhost:5432/vessel"

    # Модели
    embed_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    # Параметры поиска
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rrf_top_k: int = 20     # кандидаты после RRF → реранкер
    rerank_top_n: int = 6   # финальных чанков в контекст
    rrf_k: int = 60         # стандартная константа RRF

    # Загрузчик
    embed_batch_size: int = 8
    prefix_cache_path: str = ".prefix_cache.json"
    chunks_path: str = "chunks.json"


settings = Settings()
