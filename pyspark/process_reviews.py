"""
Phase 1a — PySpark Reviews Processing Job
E-Commerce GenAI Pipeline

Reads raw reviews.csv (Amazon Echo 2 dataset),
cleans and transforms it, saves as Parquet.

Actual columns in dataset:
Pageurl, Title, Review Text, Review Color, User Verified,
Review Date, Review Useful Count, Configuration Text, Rating, Declaration Text

Run inside Spark container:
    spark-submit /opt/spark_jobs/process_reviews.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, lower, current_timestamp,
    monotonically_increasing_id, when,
    to_date, regexp_replace
)
import sys

# ── 1. Spark Session ──────────────────────────────────────────────────────────

spark = SparkSession.builder \
    .appName("EcommerceReviewsProcessing") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("Phase 1a: Reviews Processing Job Started")
print("=" * 60)

# ── 2. Read Raw CSV ───────────────────────────────────────────────────────────

INPUT_PATH  = "/opt/data/reviews.csv"
OUTPUT_PATH = "/opt/output/reviews_clean"

print(f"\n[1/5] Reading raw CSV from: {INPUT_PATH}")

try:
    df_raw = spark.read.csv(
        INPUT_PATH,
        header=True,
        inferSchema=True,
        multiLine=True,
        escape='"'
    )
except Exception as e:
    print(f"\nERROR: Could not read {INPUT_PATH}")
    print(f"Details: {e}")
    sys.exit(1)

print(f"   Raw row count : {df_raw.count()}")
print(f"   Columns found : {df_raw.columns}")
df_raw.show(3, truncate=80)

# ── 3. Rename columns to standard schema ─────────────────────────────────────

print("\n[2/5] Renaming columns to standard schema...")

df_renamed = df_raw \
    .withColumnRenamed("Title",                "review_title") \
    .withColumnRenamed("Review Text",          "review_text") \
    .withColumnRenamed("Review Color",         "product_variant") \
    .withColumnRenamed("User Verified",        "is_verified") \
    .withColumnRenamed("Review Date",          "review_date_raw") \
    .withColumnRenamed("Review Useful Count",  "helpful_votes_raw") \
    .withColumnRenamed("Configuration Text",   "product_name") \
    .withColumnRenamed("Rating",               "rating_raw") \
    .withColumnRenamed("Declaration Text",     "declaration") \
    .withColumnRenamed("Pageurl",              "page_url")

# ── 4. Clean & Transform ──────────────────────────────────────────────────────

print("\n[3/5] Applying transformations...")

df_clean = df_renamed \
    .withColumn("review_id",
        monotonically_increasing_id().cast("string")) \
    .withColumn("product_id",
        lower(trim(col("product_name")))) \
    .withColumn("review_text",
        trim(col("review_text"))) \
    .withColumn("review_title",
        trim(col("review_title"))) \
    .withColumn("product_variant",
        lower(trim(col("product_variant")))) \
    .withColumn("is_verified",
        when(lower(trim(col("is_verified"))) == "verified purchase", True)
        .otherwise(False)) \
    .withColumn("rating",
        col("rating_raw").cast("integer")) \
    .withColumn("rating",
        when(col("rating") < 1, None)
        .when(col("rating") > 5, None)
        .otherwise(col("rating"))) \
    .withColumn("helpful_votes",
        regexp_replace(col("helpful_votes_raw").cast("string"), "[^0-9]", "")
        .cast("integer")) \
    .withColumn("review_date",
        to_date(col("review_date_raw"), "M/d/yyyy")) \
    .withColumn("ingestion_ts", current_timestamp()) \
    .filter(col("review_text").isNotNull()) \
    .filter(col("rating").isNotNull()) \
    .filter(col("product_id").isNotNull()) \
    .dropDuplicates(["review_text", "product_id"]) \
    .select(
        "review_id",
        "product_id",
        "review_title",
        "review_text",
        "rating",
        "is_verified",
        "product_variant",
        "helpful_votes",
        "review_date",
        "ingestion_ts"
    )

# ── 5. Stats ──────────────────────────────────────────────────────────────────

print("\n[4/5] Transformation summary:")
clean_count = df_clean.count()
raw_count   = df_raw.count()

print(f"   Raw rows:     {raw_count}")
print(f"   Clean rows:   {clean_count}")
print(f"   Dropped:      {raw_count - clean_count}")

print("\n   Clean sample:")
df_clean.show(5, truncate=60)

print("\n   Rating distribution:")
df_clean.groupBy("rating").count().orderBy("rating").show()

print("\n   Products found:")
df_clean.groupBy("product_id").count().orderBy("count", ascending=False).show()

# ── 6. Write Parquet ──────────────────────────────────────────────────────────

print(f"\n[5/5] Writing Parquet to: {OUTPUT_PATH}")

df_clean.coalesce(1).write \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(OUTPUT_PATH)

print(f"\n{'=' * 60}")
print(f"Reviews processing complete!")
print(f"  Output: {clean_count} rows → {OUTPUT_PATH}")
print(f"{'=' * 60}")

spark.stop()