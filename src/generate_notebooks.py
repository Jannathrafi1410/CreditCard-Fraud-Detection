"""
Generates academic Jupyter Notebooks for:
1. fraud_detection_eda.ipynb
2. pyspark_fraud_pipeline.ipynb
"""

import os
import json


def create_eda_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Credit Card Fraud Detection — Exploratory Data Analysis (EDA)\n",
                    "**Project**: Credit Card Fraud Detection Using Machine Learning and Big Data Analytics  \n",
                    "**Dataset**: Kaggle ULB Benchmark (`creditcard.csv`)  \n",
                    "**Author**: B.Tech Data Science Academic Project  \n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Environment Setup & Library Imports"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Visualization styling\n",
                    "sns.set_theme(style='whitegrid', palette='muted')\n",
                    "plt.rcParams['figure.figsize'] = (10, 6)\n",
                    "print('Libraries successfully imported!')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Ingesting the Dataset\n",
                    "We load the transaction dataset and examine its initial dimensions and column structure."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [],
                "source": [
                    "data_path = '../data/raw/creditcard.csv'\n",
                    "if not os.path.exists(data_path):\n",
                    "    data_path = '../data/sample/sample_transactions.csv'\n",
                    "\n",
                    "df = pd.read_csv(data_path)\n",
                    "print(f'Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns')\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Investigating Extreme Class Imbalance\n",
                    "Credit card fraud is rare. Detecting fraudulent transactions without overwhelming human fraud analysts with false alarms is the primary challenge."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [],
                "source": [
                    "class_counts = df['Class'].value_counts()\n",
                    "fraud_pct = (class_counts[1] / len(df)) * 100\n",
                    "legit_pct = (class_counts[0] / len(df)) * 100\n",
                    "\n",
                    "print(f'Legitimate Transactions (0): {class_counts[0]:,} ({legit_pct:.4f}%)')\n",
                    "print(f'Fraudulent Transactions (1): {class_counts[1]:,} ({fraud_pct:.4f}%)')\n",
                    "print(f'Imbalance Ratio: 1 : {class_counts[0] // class_counts[1]}')\n",
                    "\n",
                    "plt.figure(figsize=(7, 4))\n",
                    "sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette=['#3b82f6', '#ef4444'], legend=False)\n",
                    "plt.yscale('log')\n",
                    "plt.title('Class Distribution (Log Scale)', fontsize=14, fontweight='bold')\n",
                    "plt.xlabel('Class (0: Legit, 1: Fraud)')\n",
                    "plt.ylabel('Count (Log10)')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Transaction Amount Profiling\n",
                    "We analyze how transaction amounts differ between legitimate and fraudulent activities."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('--- Amount Summary Statistics by Class ---')\n",
                    "display(df.groupby('Class')['Amount'].describe().T)\n",
                    "\n",
                    "plt.figure(figsize=(8, 4))\n",
                    "sns.boxplot(x='Class', y='Amount', data=df, hue='Class', palette=['#93c5fd', '#fca5a5'], showfliers=False, legend=False)\n",
                    "plt.title('Transaction Amount by Class (Excluding Outliers)', fontsize=14, fontweight='bold')\n",
                    "plt.xticks([0, 1], ['Legitimate (0)', 'Fraudulent (1)'])\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Temporal Patterns Across 48 Hours\n",
                    "The `Time` column records elapsed seconds. We examine whether fraud displays temporal periodicity."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 5,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10, 4))\n",
                    "sns.kdeplot(df[df['Class'] == 0]['Time'], label='Legitimate (0)', fill=True, color='#2563eb', alpha=0.3)\n",
                    "sns.kdeplot(df[df['Class'] == 1]['Time'], label='Fraudulent (1)', fill=True, color='#dc2626', alpha=0.4)\n",
                    "plt.title('Transaction Temporal Density Across 48 Elapsed Hours', fontsize=14, fontweight='bold')\n",
                    "plt.xlabel('Time (Elapsed Seconds)')\n",
                    "plt.legend()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Feature Correlation with Target Class\n",
                    "We examine the strongest linear correlations with fraud."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 6,
                "metadata": {},
                "outputs": [],
                "source": [
                    "corr = df.corr()\n",
                    "top_corr = corr['Class'].abs().sort_values(ascending=False).head(15)\n",
                    "print('Top Correlated Features with Class:')\n",
                    "print(top_corr)\n",
                    "\n",
                    "plt.figure(figsize=(10, 8))\n",
                    "sns.heatmap(df[top_corr.index].corr(), cmap='coolwarm', annot=True, fmt='.2f', vmin=-1, vmax=1)\n",
                    "plt.title('Top 15 Features Correlation Heatmap', fontsize=14, fontweight='bold')\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python", "version": "3.11"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/fraud_detection_eda.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Created notebooks/fraud_detection_eda.ipynb")


def create_spark_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Big Data Processing Pipeline Using Apache Spark & PySpark\n",
                    "**Project**: Credit Card Fraud Detection Using Machine Learning and Big Data Analytics  \n",
                    "**Framework**: Apache Spark 4.2.0, Hadoop HDFS architecture  \n",
                    "**Objective**: Distributed data ingestion, hourly fraud velocity aggregation, and amount risk profiling.\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Initialize Distributed SparkSession"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pyspark.sql import SparkSession\n",
                    "from pyspark.sql.functions import col, floor, count, sum as spark_sum, avg, round as spark_round, when\n",
                    "\n",
                    "spark = SparkSession.builder \\\n",
                    "    .appName('CreditCardFraudNotebook') \\\n",
                    "    .master('local[*]') \\\n",
                    "    .config('spark.driver.memory', '2g') \\\n",
                    "    .getOrCreate()\n",
                    "\n",
                    "spark.sparkContext.setLogLevel('ERROR')\n",
                    "print(f'SparkSession initialized. Version: {spark.version}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Ingest Dataset into Distributed DataFrame"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [],
                "source": [
                    "data_path = '../data/raw/creditcard.csv'\n",
                    "df_spark = spark.read.option('header', 'true').option('inferSchema', 'true').csv(data_path)\n",
                    "print(f'Total Ingested Records: {df_spark.count():,}')\n",
                    "df_spark.printSchema()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Distributed Transformations: Hourly Velocity and Risk Buckets"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Feature engineering in Spark\n",
                    "df_enhanced = df_spark.withColumn('Elapsed_Hour', floor(col('Time') / 3600)) \\\n",
                    "                      .withColumn('Amount_Bracket',\n",
                    "                                  when(col('Amount') < 20.0, '1. Micro ($0-$20)')\n",
                    "                                  .when((col('Amount') >= 20.0) & (col('Amount') < 100.0), '2. Low ($20-$100)')\n",
                    "                                  .when((col('Amount') >= 100.0) & (col('Amount') < 500.0), '3. Medium ($100-$500)')\n",
                    "                                  .otherwise('4. High ($500+)'))\n",
                    "\n",
                    "df_enhanced.select('Time', 'Elapsed_Hour', 'Amount', 'Amount_Bracket', 'Class').show(5)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Distributed Aggregations\n",
                    "Computing fraud rates across monetary brackets using Spark Catalyst Optimizer."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [],
                "source": [
                    "bracket_summary = df_enhanced.groupBy('Amount_Bracket').agg(\n",
                    "    count('Class').alias('Total_Transactions'),\n",
                    "    spark_sum('Class').alias('Fraud_Transactions'),\n",
                    "    spark_round(avg('Class') * 100, 4).alias('Fraud_Rate_Pct')\n",
                    ").orderBy('Amount_Bracket')\n",
                    "\n",
                    "bracket_summary.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Stop Spark Session"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 5,
                "metadata": {},
                "outputs": [],
                "source": [
                    "spark.stop()\n",
                    "print('SparkSession successfully terminated.')"
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python", "version": "3.11"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open("notebooks/pyspark_fraud_pipeline.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Created notebooks/pyspark_fraud_pipeline.ipynb")


if __name__ == "__main__":
    create_eda_notebook()
    create_spark_notebook()
