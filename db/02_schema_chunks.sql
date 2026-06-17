-- v2: column-level schema chunks for vector-routed retrieval.
-- Run once against the existing container — does not auto-run (this file
-- isn't mounted into the initdb directory).
--
-- One row per column across the user-facing tables. The agent's RAG
-- retrieval step queries this table to fetch only the columns relevant
-- to a given question, instead of stuffing the whole schema into every
-- prompt.

CREATE TABLE IF NOT EXISTS docs.schema_chunks (
    id              SERIAL PRIMARY KEY,
    schema_name     TEXT NOT NULL,
    table_name      TEXT NOT NULL,
    column_name     TEXT NOT NULL,
    data_type       TEXT,
    description     TEXT,
    sample_values   TEXT,
    embedding       VECTOR(1024),
    UNIQUE (schema_name, table_name, column_name)
);

-- HNSW index on cosine distance — the modern ANN index in pgvector.
CREATE INDEX IF NOT EXISTS idx_schema_chunks_embedding
    ON docs.schema_chunks
    USING hnsw (embedding vector_cosine_ops);
