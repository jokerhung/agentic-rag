from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

embed_models = []
chat_models = []

for m in client.models.list():
    actions = m.supported_actions or []
    if "embedContent" in actions:
        embed_models.append(m)
    elif any(a in actions for a in ("generateContent", "streamGenerateContent")):
        chat_models.append(m)

print("=== EMBEDDING MODELS ===")
for m in embed_models:
    print(f"  {m.name}")

print(f"\n=== CHAT MODELS ({len(chat_models)}) ===")
for m in chat_models:
    print(f"  {m.name}")
