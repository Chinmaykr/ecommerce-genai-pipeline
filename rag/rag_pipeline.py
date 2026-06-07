"""
Phase 3 — RAG Pipeline
E-Commerce GenAI Pipeline

Flow:
  User question
    → embed question (sentence-transformers)
    → search ChromaDB (find top 5 relevant reviews)
    → build prompt (reviews as context)
    → call Gemini (generate grounded answer)
    → return answer + reviews used
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_PATH    = "../data/chromadb"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TOP_K_RESULTS  = 5   # how many reviews to retrieve per query

# ── Load model + ChromaDB once (reused for every query) ──────────────────────

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to ChromaDB...")
client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("reviews")
print(f"ChromaDB ready — {collection.count()} reviews indexed")

print("Connecting to Groq...")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("Groq ready\n")

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful e-commerce product assistant for Amazon Echo Dot.
Answer the user's question using ONLY the customer reviews provided below.
If the answer is not in the provided reviews, say "I don't have enough information from the reviews to answer that."
Do not make up features, prices, or facts.
Always mention specific details from the reviews in your answer.
Keep your answer concise — 3 to 5 sentences."""

# ── Core RAG function ─────────────────────────────────────────────────────────

def ask(user_question: str) -> dict:
    """
    Full RAG pipeline:
    1. Embed user question
    2. Search ChromaDB for top 5 relevant reviews
    3. Build prompt with retrieved reviews as context
    4. Call Gemini → get grounded answer
    5. Return answer + reviews used
    """

    print(f"\n{'='*60}")
    print(f"Question: {user_question}")
    print(f"{'='*60}")

    # ── Step 1: Embed the user question ──────────────────────────
    print("\n[1/4] Embedding question...")
    query_vector = model.encode([user_question])
    print(f"   Question → {len(query_vector[0])}-dim vector")

    # ── Step 2: Search ChromaDB ───────────────────────────────────
    print(f"\n[2/4] Searching ChromaDB for top {TOP_K_RESULTS} reviews...")
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=TOP_K_RESULTS
    )

    # Extract retrieved reviews
    retrieved_docs  = results['documents'][0]   # review texts
    retrieved_metas = results['metadatas'][0]   # rating, product_id etc
    retrieved_ids   = results['ids'][0]         # review ids

    print(f"   Found {len(retrieved_docs)} relevant reviews:")
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
        print(f"   [{i+1}] {meta['rating']}★ — {doc[:80]}...")

    # ── Step 3: Build prompt with context ─────────────────────────
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

    # ── Step 4: Call Gemini ───────────────────────────────────────
    print("\n[4/4] Calling groc API...")
    response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=500
)
    answer = response.choices[0].message.content.strip()

    # ── Step 5: Return structured result ─────────────────────────
    result = {
        "question"      : user_question,
        "answer"        : answer,
        "reviews_used"  : [
            {
                "id"     : rid,
                "rating" : meta['rating'],
                "title"  : meta.get('review_title', ''),
                "text"   : doc[:200]
            }
            for rid, doc, meta in zip(retrieved_ids, retrieved_docs, retrieved_metas)
        ]
    }

    print(f"\n{'='*60}")
    print(f"ANSWER:\n{answer}")
    print(f"{'='*60}")

    return result


# ── Test the pipeline ─────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_questions = [
        "Is the Echo Dot good for playing music?",
        "How easy is it to set up the Echo Dot?",
        "What are the common complaints about Echo Dot?",
        "Is the sound quality good enough for a bedroom?"
    ]

    for question in test_questions:
        result = ask(question)
        print(f"\nReviews used: {len(result['reviews_used'])}")
        input("\nPress Enter for next question...")
