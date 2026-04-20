"""
Kiểm tra dữ liệu Supabase và tự fix RPC schema cache.

Usage:
    python check_db.py            # kiểm tra
    python check_db.py --fix-rpc  # reload PostgREST schema rồi kiểm tra lại
"""
import os
import sys
from dotenv import load_dotenv
from supabase import create_client
from google import genai
from google.genai import types as genai_types

load_dotenv()

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
FIX_RPC = "--fix-rpc" in sys.argv

# 1. Kiểm tra số chunks
rows = supabase.table("knowledge_chunks").select("chunk_id, title").execute()
print(f"Chunks trong DB: {len(rows.data)}")
for r in rows.data:
    print(f"  {r['chunk_id']} | {r['title']}")

# 2. Kiểm tra embedding NULL
null_check = supabase.table("knowledge_chunks").select("chunk_id").is_("embedding", "null").execute()
print(f"\nEmbedding NULL: {len(null_check.data)} chunk(s)")

# 3. Reload PostgREST schema cache nếu có --fix-rpc
if FIX_RPC:
    print("\n--- Reload PostgREST schema cache ---")
    try:
        supabase.rpc("reload_pgrst_schema", {}).execute()
        print("OK — gọi reload_pgrst_schema thành công")
    except Exception:
        print("Hàm reload_pgrst_schema chưa tồn tại.")
        print("Chạy SQL này trong Supabase SQL Editor:\n")
        print("  CREATE OR REPLACE FUNCTION reload_pgrst_schema()")
        print("  RETURNS void LANGUAGE sql SECURITY DEFINER AS $$")
        print("    NOTIFY pgrst, 'reload schema';")
        print("  $$;")
        print()
        print("Sau đó chạy lại: python check_db.py --fix-rpc")
        sys.exit(0)

# 4. Embed query và test RPC
print("\n--- Test RPC match_knowledge_chunks ---")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents="các loại phòng",
    config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768),
)
vector_str = "[" + ",".join(str(v) for v in result.embeddings[0].values) + "]"

rpc_result = supabase.rpc("match_knowledge_chunks", {
    "query_embedding": list(result.embeddings[0].values),
    "match_threshold": -1.0,
    "match_count": 5,
}).execute()

if rpc_result.data:
    print(f"RPC OK — {len(rpc_result.data)} chunk(s):")
    for r in rpc_result.data:
        print(f"  {r['chunk_id']} | {r['title']} | similarity={r['similarity']:.3f}")
else:
    print("RPC trả về rỗng.")
    if not FIX_RPC:
        print("→ Thử: python check_db.py --fix-rpc")

# 5. Test RPC đơn giản không có parameter
print("\n--- Test count_chunks() RPC ---")
count_result = supabase.rpc("count_chunks", {}).execute()
print(f"count_chunks: {count_result.data}")

# 6. Test PostgREST có pass float8[] đúng không
print("\n--- Test float8[] parameter ---")
arr_result = supabase.rpc("test_array_len", {
    "arr": list(result.embeddings[0].values)
}).execute()
print(f"test_array_len (expect 768): {arr_result.data}")

# 7. Test float8[]::vector cast
print("\n--- Test float8[]::vector cast ---")
cast_result = supabase.rpc("test_vec_cast", {
    "arr": list(result.embeddings[0].values)
}).execute()
print(f"test_vec_cast (expect 768): {cast_result.data}")

# Progressive tests
arr = list(result.embeddings[0].values)
for fn in ["test_a", "test_b", "test_c"]:
    r = supabase.rpc(fn, {"arr": arr}).execute()
    print(f"{fn}: {len(r.data)} rows — {r.data[:1]}")

# 8. Test vector search trực tiếp với table
print("\n--- Test vector search in function ---")
search_result = supabase.rpc("test_vec_search", {
    "arr": list(result.embeddings[0].values)
}).execute()
print(f"test_vec_search: {search_result.data}")

# 6. Raw HTTP request để xem response thật
print("\n--- Raw HTTP request ---")
import httpx, json
embedding_list = list(result.embeddings[0].values)
resp = httpx.post(
    f"{os.environ['SUPABASE_URL']}/rest/v1/rpc/match_knowledge_chunks",
    headers={
        "apikey": os.environ["SUPABASE_SERVICE_KEY"],
        "Authorization": f"Bearer {os.environ['SUPABASE_SERVICE_KEY']}",
        "Content-Type": "application/json",
    },
    content=json.dumps({
        "query_embedding": embedding_list,
        "match_threshold": -1.0,
        "match_count": 3,
    }),
    timeout=30,
)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:300]}")

# 5. Kiểm tra tất cả versions của function
print("\n--- Kiểm tra function signatures trong DB ---")
print("Chạy SQL này trong Supabase SQL Editor:\n")
print("  SELECT proname, pg_get_function_arguments(oid), pg_get_function_result(oid)")
print("  FROM pg_proc")
print("  WHERE proname = 'match_knowledge_chunks';")
