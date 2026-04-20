"""Kiểm tra dữ liệu trong Supabase."""
import os
from dotenv import load_dotenv
from supabase import create_client
from google import genai
from google.genai import types as genai_types

load_dotenv()

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

# 1. Kiểm tra số chunks
rows = supabase.table("knowledge_chunks").select("chunk_id, title").execute()
print(f"Tổng số chunks trong DB: {len(rows.data)}")
for r in rows.data:
    print(f"  {r['chunk_id']} | {r['title']}")

# 2. Kiểm tra embedding NULL
null_check = supabase.table("knowledge_chunks").select("chunk_id").is_("embedding", "null").execute()
print(f"\nChunks có embedding NULL: {len(null_check.data)}")
for r in null_check.data:
    print(f"  {r['chunk_id']}")

# 3. Test fetch embedding về Python
sample = supabase.table("knowledge_chunks").select("chunk_id, embedding").limit(1).execute()
print(f"\nEmbedding type khi fetch: {type(sample.data[0]['embedding'])}")
print(f"Preview: {str(sample.data[0]['embedding'])[:80]}")

# 4. Embed query thật và gọi RPC
print("\n--- Test match_knowledge_chunks RPC ---")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents="các loại phòng",
    config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768),
)
embedding = result.embeddings[0].values
print(f"Query embedding type : {type(embedding)}")
print(f"Query embedding dims : {len(embedding)}")

vector_str = "[" + ",".join(str(v) for v in embedding) + "]"
print(f"Vector string preview: {vector_str[:60]}...")

# Gọi RPC với threshold = -1 để lấy tất cả
rpc_result = supabase.rpc("match_knowledge_chunks", {
    "query_embedding": vector_str,
    "match_threshold": -1.0,
    "match_count": 5,
}).execute()

print(f"RPC full response: {rpc_result}")
print(f"RPC trả về {len(rpc_result.data)} row(s):")
for r in rpc_result.data:
    print(f"  {r['chunk_id']} | similarity={r['similarity']:.4f}")

# Test gọi thẳng không qua function
print("\n--- Test direct select (không qua function) ---")
direct = supabase.table("knowledge_chunks").select("chunk_id, title").limit(3).execute()
print(f"Direct select: {direct.data}")
