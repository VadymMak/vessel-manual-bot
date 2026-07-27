-- migrations/001_init.sql
-- PostgreSQL 16 + pgvector ≥ 0.7.0
-- Запуск: psql -d vessel -f migrations/001_init.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- ─── Documents ────────────────────────────────────────────────────────────────
-- Один документ = один PDF. Чанки ссылаются на документ через doc_id.

CREATE TABLE documents (
    id          SERIAL      PRIMARY KEY,
    filename    TEXT        NOT NULL UNIQUE,   -- 'SEBU7844-37.pdf'
    title       TEXT,                          -- 'Operation and Maintenance Manual'
    page_count  INT,
    indexed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ─── Chunks ───────────────────────────────────────────────────────────────────
-- Каждая строка = один объект Chunk из ingestion/chunker.py.
-- RAG-поля: context_prefix, embedding_dense, embedding_sparse.

CREATE TABLE chunks (
    id                SERIAL      PRIMARY KEY,
    doc_id            INT         NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Поля dataclass Chunk (вербатим)
    heading           TEXT        NOT NULL,
    icode             TEXT,
    section           TEXT,
    page_start        INT         NOT NULL,
    page_end          INT         NOT NULL,
    chunk_type        TEXT        NOT NULL,     -- procedure | table | reference
    content           TEXT        NOT NULL,

    -- Массивы
    smcs_codes        TEXT[]      NOT NULL DEFAULT '{}',
    part_numbers      TEXT[]      NOT NULL DEFAULT '{}',
    illustrations     TEXT[]      NOT NULL DEFAULT '{}',
    applicable_models TEXT[]      NOT NULL DEFAULT '{}',

    -- Скалярные метаданные
    -- control_module — КРИТИЧЕСКИ ВАЖЕН: ADEM II ≠ ADEM III (правило 7 CLAUDE.md).
    -- Фильтр по этому полю идёт в SQL WHERE, а не в промпт.
    control_module       TEXT,                  -- 'ADEM II' | 'ADEM III' | NULL
    maintenance_interval TEXT,                  -- резерв для этапа 5
    step_count           INT  NOT NULL DEFAULT 0,
    part_index           INT  NOT NULL DEFAULT 0,  -- 0 = процедура целиком
    part_total           INT  NOT NULL DEFAULT 1,

    -- Блоки безопасности: [{type: 'WARNING'|'NOTICE', text: '...'}]
    safety_blocks     JSONB       NOT NULL DEFAULT '[]',
    has_warning       BOOLEAN     NOT NULL DEFAULT FALSE,

    -- RAG-специфичные поля
    -- context_prefix: 1-2 предложения от gpt-4o-mini («фрагмент из ...»).
    -- Эмбеддится: context_prefix + "\n\n" + content.
    context_prefix    TEXT,
    -- dense: bge-m3, 1024 измерения, L2-нормирован (косинус через <=>)
    embedding_dense   VECTOR(1024),
    -- sparse: bge-m3 lexical weights, vocab XLM-RoBERTa = 250002 токенов
    -- inner product через <#> (pgvector возвращает отрицательное значение)
    embedding_sparse  SPARSEVEC(250002),

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ─── Индексы ─────────────────────────────────────────────────────────────────

-- HNSW для dense ANN (косинусное расстояние).
-- m=16, ef_construction=64 — хорошее качество при малом объёме (<10k чанков).
CREATE INDEX chunks_dense_hnsw ON chunks
    USING hnsw (embedding_dense vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- SPARSEVEC: HNSW поддерживается (sparsevec_ip_ops, pgvector ≥ 0.7.0), IVFFlat — нет.
-- Ограничение HNSW: не более 1000 ненулевых элементов на вектор (тип хранит до 16000).
-- У bge-m3 число ненулевых ≈ числу уникальных токенов чанка: медиана ~300, самые
-- длинные чанки могут давать 800-1200+ — то есть на масштабе HNSW упрётся в лимит.
-- При 164 чанках sequential scan <#> даёт <50 мс — индекс не нужен.
-- При росте до 100k: обрезать sparse до топ-1000 весов per chunk, затем создать HNSW.

-- GIN для фильтрации массивов оператором &&
CREATE INDEX chunks_models_gin ON chunks USING gin (applicable_models);
CREATE INDEX chunks_smcs_gin   ON chunks USING gin (smcs_codes);
CREATE INDEX chunks_parts_gin  ON chunks USING gin (part_numbers);

-- B-tree для скалярных фильтров (правило 7: WHERE в SQL, не в промпте)
CREATE INDEX chunks_control_module_idx     ON chunks (control_module)
    WHERE control_module IS NOT NULL;
CREATE INDEX chunks_maintenance_interval_idx ON chunks (maintenance_interval)
    WHERE maintenance_interval IS NOT NULL;
CREATE INDEX chunks_chunk_type_idx         ON chunks (chunk_type);
CREATE INDEX chunks_doc_id_idx             ON chunks (doc_id);

-- ─── Комментарии ─────────────────────────────────────────────────────────────

COMMENT ON TABLE documents IS
    'Один PDF-документ. Все чанки принадлежат документу через doc_id.';

COMMENT ON TABLE chunks IS
    'Единица индексации: одна процедура или её вариант (Chunk из chunker.py). '
    'Грубо: 1 чанк = 1 процедура из O&M Manual. '
    'control_module фильтруется в WHERE (ADEM II ≠ ADEM III).';

COMMENT ON COLUMN chunks.embedding_dense IS
    'bge-m3 dense, dim=1024, L2-нормирован. Поиск: ORDER BY embedding_dense <=> $query.';

COMMENT ON COLUMN chunks.embedding_sparse IS
    'bge-m3 sparse (lexical weights), dim=250002 (XLM-RoBERTa vocab). '
    'Поиск: ORDER BY embedding_sparse <#> $query (pgvector возвращает −dot_product).';

COMMENT ON COLUMN chunks.context_prefix IS
    'Contextual retrieval prefix от gpt-4o-mini. '
    'Эмбеддится: context_prefix || E''\n\n'' || content.';
