"""
Phase 2 — Embedding Pipeline
E-Commerce GenAI Pipeline

Reads reviews_clean Parquet, generates 384-dim embeddings
using sentence-transformers, stores in ChromaDB.

Run: python embeddings/generate_embeddings.py
"""

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

REVIEWS_PARQUET  = "../output/reviews_clean"
PRODUCTS_PARQUET = "../output/products_clean"
CHROMA_PATH      = "../data/chromadb"

# ── 1. Load Parquet files ─────────────────────────────────────────────────────

print("=" * 60)
print("Phase 2: Embedding Pipeline Started")
print("=" * 60)

print("\n[1/5] Loading Parquet files...")

df_reviews = pd.read_parquet(REVIEWS_PARQUET)
df_products = pd.read_parquet(PRODUCTS_PARQUET)

print(f"   Reviews loaded  : {len(df_reviews)} rows")
print(f"   Products loaded : {len(df_products)} rows")
print(f"\n   Reviews columns : {list(df_reviews.columns)}")
print(f"   Sample review   : {df_reviews['review_text'].iloc[0][:100]}")

# ── 2. Load embedding model ───────────────────────────────────────────────────

print("\n[2/5] Loading sentence-transformers model...")
print("   (First run downloads ~80MB — normal)")

model = SentenceTransformer('all-MiniLM-L6-v2')
print("   Model loaded: all-MiniLM-L6-v2 (384 dimensions)")

# ── 3. Prepare texts for embedding ───────────────────────────────────────────

print("\n[3/5] Preparing texts for embedding...")

# Combine review title + review text for richer embedding
# This is the exact pattern from the architecture doc:
# "title + description + top reviews concatenated"

df_reviews['embed_text'] = (
    df_reviews['review_title'].fillna('') + ' ' +
    df_reviews['review_text'].fillna('')
).str.strip()

texts      = df_reviews['embed_text'].tolist()
ids        = df_reviews['review_id'].astype(str).tolist()
ratings    = df_reviews['rating'].fillna(0).astype(int).tolist()
product_ids = df_reviews['product_id'].tolist()
review_texts = df_reviews['review_text'].fillna('').tolist()
review_titles = df_reviews['review_title'].fillna('').tolist()

print(f"   Texts prepared  : {len(texts)}")
print(f"   Sample text     : {texts[0][:120]}")

# ── 4. Generate embeddings ────────────────────────────────────────────────────

print(f"\n[4/5] Generating embeddings (batch_size=64)...")
print("   This takes 1-3 minutes on first run...")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
)

print(f"\n   Embeddings generated : {len(embeddings)}")
print(f"   Dimensions per vector: {len(embeddings[0])}")

# ── 5. Store in ChromaDB ──────────────────────────────────────────────────────

print(f"\n[5/5] Storing in ChromaDB at: {CHROMA_PATH}")

os.makedirs(CHROMA_PATH, exist_ok=True)

client     = chromadb.PersistentClient(path=CHROMA_PATH)

# Delete collection if exists (clean run)
try:
    client.delete_collection("reviews")
    print("   Deleted existing collection for fresh run")
except:
    pass

collection = client.create_collection(
    name="reviews",
    metadata={"hnsw:space": "cosine"}
)

# Store in batches of 500 to avoid memory issues
BATCH_SIZE = 500
total      = len(embeddings)

for i in range(0, total, BATCH_SIZE):
    end = min(i + BATCH_SIZE, total)

    collection.add(
        embeddings = embeddings[i:end],
        ids        = ids[i:end],
        documents  = review_texts[i:end],
        metadatas  = [
            {
                "product_id"   : product_ids[j],
                "rating"       : ratings[j],
                "review_title" : review_titles[j]
            }
            for j in range(i, end)
        ]
    )
    print(f"   Stored batch {i//BATCH_SIZE + 1}: records {i} to {end}")

print(f"\n   Total stored in ChromaDB: {collection.count()}")

# ── 6. Quick test search ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Quick test — semantic search on ChromaDB")
print("=" * 60)

test_queries = [
    "good sound quality",
    "easy to set up",
    "not working properly"
]

for query in test_queries:
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    print(f"\nQuery: '{query}'")
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"  [{meta['rating']}★] {doc[:100]}...")

print(f"\n{'=' * 60}")
print("Phase 2 complete! ChromaDB is ready for RAG.")
print(f"{'=' * 60}")
