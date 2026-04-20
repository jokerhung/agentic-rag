-- Enable pgvector extension (run once per database)
create extension if not exists vector;

-- Knowledge chunks table
create table if not exists knowledge_chunks (
    id          bigserial primary key,
    chunk_id    text        not null unique,   -- e.g. "CHUNK_01"
    title       text        not null,
    tags        text[]      not null default '{}',
    content     text        not null,
    source      text        not null default 'suoikim2_knowledge.md',
    embedding   vector(768),                  -- Gemini gemini-embedding-2-preview (768 dims)
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

-- Index for fast ANN search (cosine distance)
create index if not exists knowledge_chunks_embedding_idx
    on knowledge_chunks
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 10);

-- Match function for RAG retrieval
create or replace function match_knowledge_chunks(
    query_embedding text,
    match_threshold float   default 0.0,
    match_count     int     default 5
)
returns table (
    id          bigint,
    chunk_id    text,
    title       text,
    tags        text[],
    content     text,
    similarity  float
)
language sql stable
as $$
    select
        id,
        chunk_id,
        title,
        tags,
        content,
        1 - (embedding <=> query_embedding::vector) as similarity
    from knowledge_chunks
    where 1 - (embedding <=> query_embedding::vector) > match_threshold
    order by embedding <=> query_embedding::vector
    limit match_count;
$$;
