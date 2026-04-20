"""
Ingest suoikim2_knowledge.md into Supabase pgvector.

Usage:
    python ingest.py                     # ingest all chunks
    python ingest.py --dry-run           # parse only, no DB write
    python ingest.py --chunk CHUNK_03    # ingest a single chunk

Requirements:
    pip install openai supabase python-dotenv
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass

from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

KNOWLEDGE_FILE = os.path.join(os.path.dirname(__file__), "suoikim2_knowledge.md")
EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_DIMS = 768
TABLE = "knowledge_chunks"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str       # "CHUNK_01"
    title: str          # "THÔNG TIN TỔNG QUAN"
    tags: list[str]     # ["giới thiệu", "thông tin chung", ...]
    content: str        # full markdown text of the chunk


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_CHUNK_HEADER = re.compile(r"^##\s+(CHUNK\s+\d+)\s+[—–-]+\s+(.+)$", re.MULTILINE)
_TAGS_LINE = re.compile(r"^\*\*tags:\*\*\s*(.+)$", re.MULTILINE)


def parse_knowledge_file(path: str) -> list[Chunk]:
    text = open(path, encoding="utf-8").read()

    # Split on the horizontal rules that separate chunks; keep the delimiters
    # so we can re-attach them when needed.  Each section starts with a
    # ## CHUNK XX header.
    sections = re.split(r"\n---\n", text)

    chunks: list[Chunk] = []
    for section in sections:
        header_match = _CHUNK_HEADER.search(section)
        if not header_match:
            continue

        raw_id = header_match.group(1).replace(" ", "_")  # "CHUNK_01"
        title = header_match.group(2).strip()

        tags_match = _TAGS_LINE.search(section)
        tags = (
            [t.strip() for t in tags_match.group(1).split(",")]
            if tags_match
            else []
        )

        # Content = everything after the first blank line following the header
        after_header = section[header_match.end():].lstrip("\n")
        # Strip the tags line from visible content
        content = _TAGS_LINE.sub("", after_header).strip()

        chunks.append(Chunk(chunk_id=raw_id, title=title, tags=tags, content=content))

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embed texts one by one using Gemini."""
    embeddings = []
    for i, text in enumerate(texts, 1):
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=EMBED_DIMS),
        )
        embeddings.append(result.embeddings[0].values)
        print(f"  [{i}/{len(texts)}] embedded ({len(result.embeddings[0].values)} dims)")
    return embeddings


# ---------------------------------------------------------------------------
# Supabase upsert
# ---------------------------------------------------------------------------

def upsert_chunks(
    supabase: Client,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    rows = [
        {
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "tags": chunk.tags,
            "content": chunk.content,
            "embedding": "[" + ",".join(str(v) for v in embedding) + "]",
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    result = (
        supabase.table(TABLE)
        .upsert(rows, on_conflict="chunk_id")
        .execute()
    )

    if hasattr(result, "error") and result.error:
        raise RuntimeError(f"Supabase upsert error: {result.error}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge chunks into Supabase pgvector")
    parser.add_argument("--dry-run", action="store_true", help="Parse only; skip embedding and DB write")
    parser.add_argument("--chunk", metavar="ID", help="Ingest a single chunk by ID, e.g. CHUNK_03")
    args = parser.parse_args()

    # Parse ----------------------------------------------------------------
    chunks = parse_knowledge_file(KNOWLEDGE_FILE)
    if not chunks:
        print("No chunks found — check the knowledge file format.")
        sys.exit(1)

    if args.chunk:
        target = args.chunk.upper().replace(" ", "_")
        chunks = [c for c in chunks if c.chunk_id == target]
        if not chunks:
            print(f"Chunk '{target}' not found. Available: {[c.chunk_id for c in parse_knowledge_file(KNOWLEDGE_FILE)]}")
            sys.exit(1)

    print(f"Parsed {len(chunks)} chunk(s):")
    for c in chunks:
        print(f"  {c.chunk_id}: {c.title} [{', '.join(c.tags[:3])}{'...' if len(c.tags) > 3 else ''}]")

    if args.dry_run:
        print("\n--dry-run: stopping before embedding / DB write.")
        return

    # Embed ----------------------------------------------------------------
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("Missing GEMINI_API_KEY in environment / .env file")
        sys.exit(1)

    gemini_client = genai.Client(api_key=gemini_key)

    print(f"\nEmbedding {len(chunks)} chunk(s) with {EMBED_MODEL}...")
    texts = [f"{c.title}\n\n{c.content}" for c in chunks]
    embeddings = embed_texts(gemini_client, texts)
    print("Embedding done.")

    # Upsert ---------------------------------------------------------------
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        print("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in environment / .env file")
        sys.exit(1)

    supabase: Client = create_client(supabase_url, supabase_key)

    print(f"Upserting {len(chunks)} row(s) into '{TABLE}'...")
    upsert_chunks(supabase, chunks, embeddings)
    print(f"Done. {len(chunks)} chunk(s) upserted successfully.")


if __name__ == "__main__":
    main()
