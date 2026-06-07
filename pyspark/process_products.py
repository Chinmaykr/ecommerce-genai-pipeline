"""
Phase 1 — PySpark Products Processing Job
E-Commerce GenAI Pipeline

Reads raw products.csv, cleans and transforms it,
saves output as Parquet files for downstream embedding + RAG use.

Run inside Spark container:
    spark-submit /opt/spark_jobs/process_products.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, trim, current_timestamp,
    when, regexp_replace
)
import sys

# ── 1. Create Spark Session ───────────────────────────────────────────────────

spark = SparkSession.builder \
    .appName("EcommerceProductsProcessing") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("Phase 1: Products Processing Job Started")
print("=" * 60)

# ── 2. Read Raw CSV ───────────────────────────────────────────────────────────

INPUT_PATH  = "/opt/data/products.csv"
OUTPUT_PATH = "/opt/output/products_clean"

print(f"\n[1/6] Reading raw CSV from: {INPUT_PATH}")

try:
    df_raw = spark.read.csv(
        INPUT_PATH,
        header=True,
        inferSchema=True,
        multiLine=True,       # handles descriptions with newlines
        escape='"'
    )
except Exception as e:
    print(f"\nERROR: Could not read {INPUT_PATH}")
    print(f"Make sure products.csv is in your data/ folder.")
    print(f"Details: {e}")
    sys.exit(1)

print(f"   Raw row count: {df_raw.count()}")
print(f"   Columns found: {df_raw.columns}")

# ── 3. Preview Raw Data ───────────────────────────────────────────────────────

print("\n[2/6] Raw data sample:")
df_raw.show(3, truncate=80)
df_raw.printSchema()

# ── 4. Rename & Select Columns ────────────────────────────────────────────────
# Handles both standard schema and Kaggle Amazon echo dataset column names
# The architecture doc schema: product_id, title, description, category,
#                               price, brand, avg_rating, review_count

print("\n[3/6] Mapping columns to standard schema...")

actual_cols = [c.lower().strip() for c in df_raw.columns]

# Build a flexible column mapping — covers common Kaggle dataset variations
col_map = {}
for c in df_raw.columns:
    cl = c.lower().strip()
    if "product_id" in cl or cl == "id" or cl == "asin":
        col_map["product_id"] = c
    elif "title" in cl or "name" in cl or "product_name" in cl:
        col_map["title"] = c
    elif "description" in cl or "about" in cl or "feature" in cl:
        col_map["description"] = c
    elif "category" in cl or "department" in cl:
        col_map["category"] = c
    elif "price" in cl:
        col_map["price"] = c
    elif "brand" in cl or "manufacturer" in cl:
        col_map["brand"] = c
    elif "rating" in cl and "count" not in cl and "num" not in cl:
        col_map["avg_rating"] = c
    elif "review_count" in cl or "num_review" in cl or "total_review" in cl:
        col_map["review_count"] = c

print(f"   Detected column mapping: {col_map}")

# Build select expressions using detected columns
# Fall back gracefully if a column is missing
select_exprs = []

select_exprs.append(
    col(col_map.get("product_id", col_map.get("title", df_raw.columns[0]))).cast("string").alias("product_id")
)
select_exprs.append(
    col(col_map.get("title", df_raw.columns[0])).cast("string").alias("title")
)
select_exprs.append(
    col(col_map.get("description", col_map.get("title", df_raw.columns[0]))).cast("string").alias("description")
)

# Optional columns — use null if not present
if "category" in col_map:
    select_exprs.append(col(col_map["category"]).cast("string").alias("category"))
else:
    select_exprs.append(col(df_raw.columns[0]).cast("string").isNull().cast("string").alias("category"))

if "price" in col_map:
    select_exprs.append(
        regexp_replace(col(col_map["price"]).cast("string"), "[^0-9.]", "").cast("float").alias("price")
    )
else:
    select_exprs.append(col(df_raw.columns[0]).cast("float").alias("price"))

if "brand" in col_map:
    select_exprs.append(col(col_map["brand"]).cast("string").alias("brand"))
else:
    select_exprs.append(col(df_raw.columns[0]).cast("string").alias("brand"))

if "avg_rating" in col_map:
    select_exprs.append(col(col_map["avg_rating"]).cast("float").alias("avg_rating"))
else:
    select_exprs.append(col(df_raw.columns[0]).cast("float").alias("avg_rating"))

if "review_count" in col_map:
    select_exprs.append(col(col_map["review_count"]).cast("integer").alias("review_count"))
else:
    select_exprs.append(col(df_raw.columns[0]).cast("integer").alias("review_count"))

df_mapped = df_raw.select(select_exprs)

# ── 5. Clean & Transform ─────────────────────────────────────────────────────

print("\n[4/6] Applying transformations...")

df_clean = df_mapped \
    .dropDuplicates(["product_id"]) \
    .filter(col("product_id").isNotNull()) \
    .filter(col("title").isNotNull()) \
    .filter(col("description").isNotNull()) \
    .withColumn("title",       lower(trim(col("title")))) \
    .withColumn("description", trim(col("description"))) \
    .withColumn("brand",       lower(trim(col("brand")))) \
    .withColumn("category",    lower(trim(col("category")))) \
    .withColumn("avg_rating",
        when(col("avg_rating") < 1.0, None)
        .when(col("avg_rating") > 5.0, None)
        .otherwise(col("avg_rating"))
    ) \
    .withColumn("price",
        when(col("price") < 0, None)
        .otherwise(col("price"))
    ) \
    .withColumn("ingestion_ts", current_timestamp())

# ── 6. Stats ──────────────────────────────────────────────────────────────────

print("\n[5/6] Transformation summary:")
clean_count = df_clean.count()
raw_count   = df_raw.count()
dropped     = raw_count - clean_count

print(f"   Raw rows:     {raw_count}")
print(f"   Clean rows:   {clean_count}")
print(f"   Dropped rows: {dropped} (nulls + duplicates)")

print("\n   Clean data sample:")
df_clean.show(5, truncate=80)
df_clean.printSchema()

print("\n   Rating distribution:")
df_clean.groupBy("avg_rating").count().orderBy("avg_rating").show()

# ── 7. Write Parquet ─────────────────────────────────────────────────────────

print(f"\n[6/6] Writing Parquet to: {OUTPUT_PATH}")

df_clean.coalesce(1).write \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(OUTPUT_PATH)

print(f"\n{'=' * 60}")
print(f"Products processing complete!")
print(f"  Input : {raw_count} rows from {INPUT_PATH}")
print(f"  Output: {clean_count} rows → {OUTPUT_PATH}")
print(f"{'=' * 60}")

spark.stop()