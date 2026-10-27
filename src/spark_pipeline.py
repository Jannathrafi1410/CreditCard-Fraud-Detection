"""
Distributed Big Data Analytics Pipeline for Credit Card Fraud Detection.
Implements scalable data ingestion, aggregation, hourly fraud velocity profiling,
and amount distribution statistics using Apache Spark / PySpark.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PySpark imports
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, floor, count, sum as spark_sum, avg, stddev, min as spark_min, max as spark_max, when, round as spark_round
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


def create_spark_session(app_name: str = "CreditCardFraudSparkPipeline", master: str = "local[*]") -> "SparkSession":
    """
    Initializes a resilient SparkSession configured for local or cluster execution.
    """
    if not PYSPARK_AVAILABLE:
        raise ImportError("PySpark is not installed. Run 'pip install pyspark' to enable Big Data processing.")

    # Configure session with sensible memory allocations
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()

    # Suppress verbose log4j warnings
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def run_spark_fraud_pipeline(input_path: str, output_dir: str = "data/processed/spark_aggregated_metrics") -> dict:
    """
    Executes distributed transformations on the transaction dataset.
    """
    print(f"\n[PySpark Pipeline] Initializing Spark Big Data Pipeline on: {input_path}")
    spark = create_spark_session()
    print(f"[PySpark Pipeline] SparkSession active (Version: {spark.version})")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # 1. Read CSV into distributed DataFrame with schema inference
    df_spark = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
    total_records = df_spark.count()
    total_columns = len(df_spark.columns)
    print(f"[PySpark Pipeline] Ingested {total_records:,} rows and {total_columns} columns into Spark DataFrame.")

    # 2. Add derived distributed features: Elapsed Hour and Amount Buckets
    df_enhanced = df_spark.withColumn("Elapsed_Hour", floor(col("Time") / 3600)) \
                          .withColumn("Hour_of_Day", floor(col("Time") / 3600) % 24) \
                          .withColumn("Amount_Bracket",
                                      when(col("Amount") < 20.0, "1. Micro ($0-$20)")
                                      .when((col("Amount") >= 20.0) & (col("Amount") < 100.0), "2. Low ($20-$100)")
                                      .when((col("Amount") >= 100.0) & (col("Amount") < 500.0), "3. Medium ($100-$500)")
                                      .otherwise("4. High ($500+)"))

    # 3. Distributed Hourly Fraud Velocity Aggregation
    hourly_stats_spark = df_enhanced.groupBy("Elapsed_Hour").agg(
        count("Class").alias("Total_Transactions"),
        spark_sum("Class").alias("Fraud_Transactions"),
        spark_round(avg("Class") * 100, 4).alias("Fraud_Percentage"),
        spark_round(avg("Amount"), 2).alias("Avg_Transaction_Amount")
    ).orderBy("Elapsed_Hour")

    hourly_df = hourly_stats_spark.toPandas()
    hourly_out_csv = os.path.join(output_dir, "spark_hourly_fraud_summary.csv")
    hourly_df.to_csv(hourly_out_csv, index=False)
    print(f"[PySpark Pipeline] Hourly statistics exported to: {hourly_out_csv}")

    # 4. Amount Statistics Grouped by Class
    amount_stats_spark = df_enhanced.groupBy("Class").agg(
        count("Amount").alias("Count"),
        spark_round(avg("Amount"), 2).alias("Mean_Amount"),
        spark_round(stddev("Amount"), 2).alias("Std_Amount"),
        spark_round(spark_min("Amount"), 2).alias("Min_Amount"),
        spark_round(spark_max("Amount"), 2).alias("Max_Amount")
    )
    amount_stats_df = amount_stats_spark.toPandas()
    amount_out_csv = os.path.join(output_dir, "spark_class_amount_metrics.csv")
    amount_stats_df.to_csv(amount_out_csv, index=False)
    print(f"[PySpark Pipeline] Amount statistics exported to: {amount_out_csv}")

    # 5. Amount Bracket Risk Profiling
    bracket_stats_spark = df_enhanced.groupBy("Amount_Bracket").agg(
        count("Class").alias("Total_Transactions"),
        spark_sum("Class").alias("Fraud_Count"),
        spark_round(avg("Class") * 100, 4).alias("Fraud_Rate_Pct")
    ).orderBy("Amount_Bracket")
    bracket_df = bracket_stats_spark.toPandas()
    bracket_out_csv = os.path.join(output_dir, "spark_amount_bracket_risk.csv")
    bracket_df.to_csv(bracket_out_csv, index=False)
    print(f"[PySpark Pipeline] Amount Bracket risk profile exported to: {bracket_out_csv}")

    # 6. Generate Hourly Fraud Trends Visual
    plt.figure(figsize=(10, 5), dpi=300)
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    ax2 = ax1.twinx()

    p1 = ax1.bar(hourly_df["Elapsed_Hour"], hourly_df["Total_Transactions"], color="#cbd5e1", alpha=0.7, label="Total Volume (left)")
    p2 = ax2.plot(hourly_df["Elapsed_Hour"], hourly_df["Fraud_Transactions"], color="#dc2626", lw=2.2, marker="o", markersize=4, label="Fraud Count (right)")

    ax1.set_xlabel("Elapsed Time (Hours from Dataset Inception)", fontsize=11)
    ax1.set_ylabel("Total Transaction Volume", color="#475569", fontsize=11)
    ax2.set_ylabel("Fraudulent Transactions Count", color="#dc2626", fontsize=11)
    plt.title("PySpark Distributed Analysis: Hourly Transaction Volume & Fraud Velocity", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    plot_path = "reports/figures/spark_hourly_fraud_trends.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[PySpark Pipeline] Hourly trend figure generated: {plot_path}")

    # Clean shutdown of Spark session
    spark.stop()
    print("[PySpark Pipeline] SparkSession successfully stopped.")

    return {
        "total_records": total_records,
        "total_hours": len(hourly_df),
        "amount_summary": amount_stats_df.to_dict(orient="records"),
        "bracket_summary": bracket_df.to_dict(orient="records")
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Fraud Analytics via PySpark")
    parser.add_argument("--input", default="data/raw/creditcard.csv", help="Path to raw transaction CSV")
    parser.add_argument("--output", default="data/processed/spark_aggregated_metrics", help="Directory for Spark outputs")
    args = parser.parse_args()

    # Use sample if raw full CSV does not exist
    input_file = args.input
    if not os.path.exists(input_file):
        sample_file = "data/sample/sample_transactions.csv"
        if os.path.exists(sample_file):
            print(f"Notice: {input_file} not found. Running Spark pipeline on {sample_file} instead.")
            input_file = sample_file
        else:
            print(f"Error: Neither {input_file} nor sample file was found.")
            sys.exit(1)

    results = run_spark_fraud_pipeline(input_file, args.output)
    print("\n--- Summary of PySpark Distributed Output ---")
    print(f"Records Processed: {results['total_records']:,}")
    print("Amount Brackets Analysis:")
    for b in results["bracket_summary"]:
        print(f"  {b['Amount_Bracket']}: {b['Total_Transactions']} trans, {b['Fraud_Count']} frauds ({b['Fraud_Rate_Pct']}%)")
