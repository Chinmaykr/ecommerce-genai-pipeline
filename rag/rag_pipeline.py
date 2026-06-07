"""
Phase 3 — RAG Pipeline
Works both locally and on Streamlit Cloud
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Smart path detection ──────────────────────────────────────────────────────
# Works locally (../data/chromadb) and on Streamlit Cloud (chromadb_store/)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if os.path.exists(os.path.join(BASE_DIR, "chromadb_store")):
    # Streamlit Cloud or bundled deployment
    CHROMA_PATH    = os.path.join(BASE_DIR, "chromadb_store")
    REVIEWS_PATH   = os.path.join(BASE_DIR, "parquet_store", "reviews_clean")
    PRODUCTS_PATH  = os.path.join(BASE_DIR, "parquet_store", "products_clean")
else:
    # Local Mac setup
    CHROMA_PATH    = os.path.join(BASE_DIR, "..", "data", "chromadb")
    REVIEWS_PATH   = os.path.join(BASE_DIR, "..", "output", "reviews_clean")
    PRODUCTS_PATH  = os.path.join(BASE_DIR, "..", "output", "products_clean")

TOP_K_RESULTS = 5

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to ChromaDB...")
client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("reviews")
print(f"ChromaDB ready — {collection.count()} reviews indexed")

print("Connecting to Groq...")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("Groq ready\n")

SYSTEM_PROMPT = """You are a helpful e-commerce product assistant for Amazon Echo Dot.
Answer the user's question using ONLY the customer reviews provided below.
If the answer is not in the provided reviews, say "I don't have enough information from the reviews to answer that."
Do not make up features, prices, or facts.
Always mention specific details from the reviews in your answer.
Keep your answer concise — 3 to 5 sentences."""

def ask(user_question: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Question: {user_question}")
    print(f"{'='*60}")

    print("\n[1/4] Embedding question...")
    query_vector = model.encode([user_question])
    print(f"   Question → {len(query_vector[0])}-dim vector")

    print(f"\n[2/4] Searching ChromaDB for top {TOP_K_RESULTS} reviews...")
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=TOP_K_RESULTS
    )

    retrieved_docs  = results['documents'][0]
    retrieved_metas = results['metadatas'][0]
    retrieved_ids   = results['ids'][0]

    print(f"   Found {len(retrieved_docs)} relevant reviews:")
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
        print(f"   [{i+1}] {meta['rating']}★ — {doc[:80]}...")

    print("\n[3/4] Building prompt with retrieved reviews...")
    context = ""
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
        context += f"""
Review {i+1}:
  Rating    : {meta['rating']} out of 5 stars
  Title     : {meta.get('review_title', 'N/A')}
  Review    : {doc}
---"""

    prompt = f"""{SYSTEM_PROMPT}

CUSTOMER REVIEWS:
{context}

USER QUESTION: {user_question}

YOUR ANSWER:"""

    print(f"   Prompt built — {len(prompt)} characters")

    print("\n[4/4] Calling Groq API...")
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    answer = response.choices[0].message.content.strip()

    result = {
        "question"    : user_question,
        "answer"      : answer,
        "reviews_used": [
            {
                "id"    : rid,
                "rating": meta['rating'],
                "title" : meta.get('review_title', ''),
                "text"  : doc[:200]
            }
            for rid, doc, meta in zip(retrieved_ids, retrieved_docs, retrieved_metas)
        ]
    }

    print(f"\n{'='*60}")
    print(f"ANSWER:\n{answer}")
    print(f"{'='*60}")

    return result


if __name__ == "__main__":
    test_questions = [
        "Is the Echo Dot good for playing music?",
        "How easy is it to set up the Echo Dot?",
    ]
    for question in test_questions:
        result = ask(question)
        print(f"\nReviews used: {len(result['reviews_used'])}")

