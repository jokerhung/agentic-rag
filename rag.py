"""
RAG Q&A — Suối Kim 2 Homestay
Dùng Gemini embeddings + Supabase pgvector + LangChain

Usage: python rag.py
"""

import math
import os
from typing import List

from dotenv import load_dotenv
from google import genai as google_genai
from google.genai import types as genai_types
from supabase import create_client, Client

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_DIMS = 768
CHAT_MODEL = "gemma-4-31b-it"


# ---------------------------------------------------------------------------
# Custom Embeddings (dùng google-genai SDK mới)
# ---------------------------------------------------------------------------

class GeminiEmbeddings(Embeddings):
    def __init__(self):
        self.client = google_genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _embed(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBED_DIMS,
            ),
        )
        return result.embeddings[0].values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


# ---------------------------------------------------------------------------
# Custom Retriever — gọi match_knowledge_chunks RPC
# ---------------------------------------------------------------------------

class SupabaseRetriever(BaseRetriever):
    supabase: Client
    embeddings: GeminiEmbeddings
    match_threshold: float = 0.0
    match_count: int = 5
    debug: bool = False

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
        return dot / norm if norm else 0.0

    def _search_via_rpc(self, vector_str: str) -> List[dict] | None:
        """Dùng pgvector ANN index — hiệu quả cho dataset lớn."""
        result = self.supabase.rpc("match_knowledge_chunks", {
            "query_embedding": vector_str,
            "match_threshold": self.match_threshold,
            "match_count": self.match_count,
        }).execute()
        # RPC trả về [] có thể do schema cache cũ — fallback nếu rỗng
        return result.data if result.data else None

    def _search_via_fetch(self, query_embedding: List[float]) -> List[tuple]:
        """Fetch toàn bộ rồi tính cosine similarity trong Python — dùng khi dataset nhỏ hoặc RPC lỗi."""
        rows = self.supabase.table("knowledge_chunks").select(
            "chunk_id, title, content, tags, embedding"
        ).execute().data

        scored = []
        for row in rows:
            stored = [float(v) for v in row["embedding"][1:-1].split(",")]
            sim = self._cosine_similarity(query_embedding, stored)
            if sim > self.match_threshold:
                scored.append((sim, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[: self.match_count]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_embedding = self.embeddings.embed_query(query)
        vector_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        # Thử RPC trước (tận dụng pgvector index cho dataset lớn)
        rpc_rows = self._search_via_rpc(vector_str)
        if rpc_rows is not None:
            if self.debug:
                print(f"\n[DEBUG] RPC: {len(rpc_rows)} chunk(s)")
                for r in rpc_rows:
                    print(f"  {r['chunk_id']} | {r['title']} | similarity={r['similarity']:.3f}")
                print()
            return [
                Document(
                    page_content=r["content"],
                    metadata={"chunk_id": r["chunk_id"], "title": r["title"], "similarity": round(r["similarity"], 3)},
                )
                for r in rpc_rows
            ]

        # Fallback: tính trong Python (dataset nhỏ hoặc RPC chưa reload schema)
        if self.debug:
            print("\n[DEBUG] RPC trả về rỗng, dùng fetch fallback")
        top = self._search_via_fetch(query_embedding)
        if self.debug:
            print(f"[DEBUG] Fetch: {len(top)} chunk(s)")
            for sim, row in top:
                print(f"  {row['chunk_id']} | {row['title']} | similarity={sim:.3f}")
            print()
        return [
            Document(
                page_content=row["content"],
                metadata={"chunk_id": row["chunk_id"], "title": row["title"], "similarity": round(sim, 3)},
            )
            for sim, row in top
        ]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Bạn là trợ lý tư vấn của Suối Kim 2 Homestay tại Mù Cang Chải, Yên Bái.
Hãy trả lời câu hỏi của khách dựa trên thông tin trong context bên dưới.
Nếu context không có thông tin cần thiết, hãy thành thật nói không biết và gợi ý khách liên hệ trực tiếp: 0329 961 420.
Trả lời bằng tiếng Việt, thân thiện, ngắn gọn và đúng trọng tâm.

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        parts.append(f"[{doc.metadata['title']} — similarity: {doc.metadata['similarity']}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

def build_chain(debug: bool = False):
    embeddings = GeminiEmbeddings()
    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )
    retriever = SupabaseRetriever(supabase=supabase, embeddings=embeddings, debug=debug)
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.3,
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import sys
    debug = "--debug" in sys.argv

    print("=" * 50)
    print("  Suối Kim 2 Homestay — Trợ lý tư vấn")
    print("  Gõ 'exit' để thoát" + (" | DEBUG ON" if debug else ""))
    print("=" * 50 + "\n")

    chain = build_chain(debug=debug)

    while True:
        try:
            question = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "thoát"):
            print("Tạm biệt!")
            break

        print("\nTrợ lý: ", end="", flush=True)
        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
