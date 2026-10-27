# Credit Card Fraud Detection Using Machine Learning & Big Data Analytics

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.2+-276DC3?logo=r&logoColor=white)](https://www.r-project.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-4.2.0-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic%20B.Tech-blue.svg)]()

An end-to-end, academic-grade Machine Learning and Big Data analytics capstone project for Credit Card Fraud Detection. The system implements a **dual-stack architecture**:
1. **Primary Academic Machine Learning Pipeline in R**: Built using RStudio, `dplyr`, `ggplot2`, `caret`, and Breiman's `randomForest`.
2. **Distributed Big Data & Interactive Serving Stack in Python**: Built using Apache Hadoop (HDFS), Apache Spark (`pyspark`), `scikit-learn`, and an interactive `streamlit` web dashboard.

---

## 📌 Project Overview & Highlights
- **Real Benchmark Dataset**: Evaluated on the ULB Kaggle dataset containing **284,807 European credit card transactions** (492 frauds, **0.1727% fraud rate**).
- **Leak-Free Preprocessing**: Applied `RobustScaler` on skewed features (`Time`, `Amount`) fitted strictly on training data alongside stratified 80/20 train/test partitioning.
- **Cost-Sensitive Learning**: Controlled extreme class imbalance using loss weighting ($w_j = \frac{N}{k \cdot n_j}$), avoiding synthetic manifold distortions from SMOTE.
- **Superior Experimental Performance**:
  - **Accuracy**: `99.937%`
  - **Precision**: `81.000%` (98.6% fewer false alarms than baseline)
  - **Recall**: `82.653%` (81 out of 98 test frauds caught)
  - **F1-Score**: `0.8182` (vs baseline `0.1142`)
  - **PR-AUC**: `0.8418` | **ROC-AUC**: `0.9614`
- **Distributed Big Data Analytics**: PySpark pipeline calculating distributed hourly velocity profiles, amount brackets, and simulating HDFS storage.
- **Interactive Multi-Module Dashboard**: Streamlit web app supporting live single-transaction evaluation, batch scoring with downloadable CSVs, and visual EDA.

---

## 📂 Project Folder Structure

```text
Credit-Card Fraud/
├── README.md                           # Master documentation and run guide
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Clean repository hygiene rules
├── config/
│   └── config.yaml                     # Centralized hyperparameters & paths
├── data/
│   ├── raw/
│   │   └── creditcard.csv              # Download locally; excluded from Git (>100 MB)
│   ├── processed/
│   │   ├── spark_aggregated_metrics/   # Distributed Spark output CSVs
│   │   └── train_r.RData               # R preprocessed training set
│   └── sample/
│       ├── sample_transactions.csv     # 500-row balanced sample for quick demos
│       ├── test_batch_transactions.csv # 50-row unlabelled test batch
│       └── scored_batch_r_output.csv   # Scored batch output from R
├── R/
│   ├── install_packages.R              # Automated R package installer
│   ├── 01_data_inspection.R           # Technical dataset audit in R
│   ├── 02_data_preprocessing.R         # Robust scaling & stratified split
│   ├── 03_exploratory_analysis.R       # Publication-quality ggplot2 charts
│   ├── 04_train_random_forest.R        # Caret / randomForest training
│   ├── 05_evaluate_model.R             # Held-out test evaluation & confusion matrix
│   └── 06_predict_transactions.R       # Batch inference on new records
├── notebooks/
│   ├── fraud_detection_eda.ipynb       # Jupyter notebook for exploratory analysis
│   └── pyspark_fraud_pipeline.ipynb    # Jupyter notebook for PySpark pipeline
├── src/
│   ├── data_loader.py                  # Ingestion, schema check, target detection
│   ├── preprocessing.py                # RobustScaler class & EDA plots
│   ├── train_model.py                  # Random Forest & Baseline Logistic Regression
│   ├── evaluate_model.py               # Metrics calculation & diagnostic plots
│   ├── predict.py                      # Single & batch inference engine
│   ├── spark_pipeline.py               # Apache Spark distributed pipeline
│   ├── hdfs_utils.py                   # HDFS command manager & local sandbox
│   ├── generate_diagrams.py            # High-res architecture & DFD generator
│   ├── generate_notebooks.py           # Notebook generator
│   └── main_pipeline.py                # Master orchestrator script
├── app/
│   └── streamlit_app.py                # 7-module interactive web dashboard
├── models/
│   ├── random_forest_model.joblib      # Serialized trained model
│   ├── baseline_logistic_regression.joblib # Baseline comparison model
│   ├── scaler.joblib                   # Serialized RobustScaler
│   ├── model_metadata.json             # Hyperparameters & feature names
│   └── r_scaler_params.RData           # Scaler medians & IQRs for R
├── reports/
│   ├── figures/                        # High-resolution PNG charts
│   │   ├── class_distribution.png
│   │   ├── amount_by_class_boxplot.png
│   │   ├── transaction_time_density.png
│   │   ├── correlation_heatmap.png
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── precision_recall_curve.png
│   │   ├── feature_importance.png
│   │   └── spark_hourly_fraud_trends.png
│   ├── metrics/
│   │   ├── dataset_inspection.json     # Full statistical inspection summary
│   │   ├── evaluation_metrics.json     # Test evaluation results & confusion matrix
│   │   └── confusion_matrix.json       # TP, TN, FP, FN breakdown
│   ├── project_report.md               # 21-section B.Tech Capstone Project Report
│   └── viva_questions.md               # 36 comprehensive viva defense Q&A
├── diagrams/
│   ├── system_architecture.png         # Dual ML + Big Data architecture
│   ├── dfd_level_0.png                 # Context Level Data Flow Diagram
│   ├── dfd_level_1.png                 # Functional Decomposition DFD
│   ├── dfd_level_2.png                 # Detailed Inference Subprocess DFD
│   └── system_flowchart.png            # End-to-end logic flowchart
└── tests/
    ├── test_data_loading.py            # Automated tests for ingestion & target
    ├── test_preprocessing.py           # Tests for scaling, leakage & shapes
    └── test_predictions.py             # Tests for inference, bounds & thresholds
```

---

## ⚙️ Hardware & Software Requirements

### Hardware
- **Processor**: Intel Core i5 / AMD Ryzen 5 or higher (minimum 4 cores).
- **RAM**: Minimum 8 GB (16 GB recommended for running PySpark JVM and Streamlit simultaneously).
- **Storage**: At least 3 GB of available disk space.

### Software Environments
- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python**: Version 3.10 or 3.11 with `pip`.
- **Java**: OpenJDK 11 or 17 (required for Apache Spark / PySpark).
- **R & RStudio**: R version 4.2+ and RStudio Desktop.

---

## 🚀 Step-by-Step Installation & Setup

### 1. Clone or Extract the Project
Open a terminal (PowerShell, Bash, or Command Prompt) and navigate to the project directory:
```bash
cd "Credit-Card Fraud"
```

### 2. Configure Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Verify Dataset Placement
Ensure that `creditcard.csv` is present in `data/raw/creditcard.csv`.
If you need to download it automatically:
```python
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Downloaded to:", path)
```
*Note: A ready-to-run 500-record representative sample is included in `data/sample/sample_transactions.csv`, allowing the entire project to run out of the box even before downloading the 144 MB raw file!*

---

## 🏃 Execution Instructions

### A. Run Master End-to-End Python Pipeline
Executes data ingestion, EDA figure generation, preprocessing, Random Forest training, evaluation, metrics export, and PySpark analytics in a single command:
```bash
python src/main_pipeline.py
```

### B. Launch the Interactive Streamlit Web Dashboard
Launch the dashboard to demonstrate real-time scoring and visual analytics:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

#### Dashboard Modules:
1. **Project Overview**: Problem statement, objectives, and system architecture.
2. **Dataset Inspection**: Full KPI metrics, feature breakdown, and privacy safeguards.
3. **Exploratory Data Analysis**: Interactive class distributions, temporal patterns, and correlation matrices.
4. **Model Performance**: Precision, Recall, F1-Score, Confusion Matrix, and ROC/PR curves.
5. **Single Transaction Prediction**: Interactive risk scoring form with demo pre-fill buttons and a dynamic probability gauge.
6. **Batch Transaction Scoring**: Drag-and-drop CSV batch upload with downloadable risk scores.
7. **Operational Fraud Monitoring**: Real-time risk exposure cards and prioritized queue review.

### C. Run the Primary Academic R Implementation
Open **RStudio**, set your working directory to the project folder, and run the R scripts sequentially:
```R
# 1. Install all dependencies
source("R/install_packages.R")

# 2. Inspect dataset
source("R/01_data_inspection.R")

# 3. Preprocess and split data
source("R/02_data_preprocessing.R")

# 4. Generate ggplot2 exploratory figures
source("R/03_exploratory_analysis.R")

# 5. Train Random Forest model with caret/randomForest
source("R/04_train_random_forest.R")

# 6. Evaluate on held-out test data
source("R/05_evaluate_model.R")

# 7. Score new transactions
source("R/06_predict_transactions.R")
```

### D. Execute the Distributed Big Data PySpark Pipeline
Run the distributed PySpark script directly:
```bash
python src/spark_pipeline.py --input data/raw/creditcard.csv --output data/processed/spark_aggregated_metrics
```

#### Native Apache Hadoop Cluster Commands (Reference for Viva/Demo):
```bash
# 1. Create HDFS input/output directories
hdfs dfs -mkdir -p /fraud_detection/input
hdfs dfs -mkdir -p /fraud_detection/output

# 2. Upload raw transaction CSV to HDFS
hdfs dfs -put -f data/raw/creditcard.csv /fraud_detection/input/

# 3. Submit Spark Job to the Hadoop YARN cluster
spark-submit \
  --master yarn \
  --deploy-mode client \
  --driver-memory 4G \
  --executor-memory 4G \
  src/spark_pipeline.py \
  --input hdfs:///fraud_detection/input/creditcard.csv \
  --output hdfs:///fraud_detection/output/
```

### E. Run Automated Unit Tests
Run the test suite to verify module integrity:
```bash
pytest tests -v
```
*(All 14 unit tests pass verifying data loading, target detection, leak-free scaling, and inference).*

---

## 📊 Measured Experimental Results

All metrics below reflect execution on the **unseen held-out test split of 56,962 transactions (98 frauds, 56,864 legitimate)**:

| Metric | Random Forest (Our Model) | Baseline (Logistic Regression) | Operational Impact |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **99.937%** | 97.549% | High overall accuracy |
| **Precision** | **81.000%** | 6.089% | **Random Forest cuts false alarms by 98.6%** |
| **Recall (Sensitivity)** | 82.653% | **91.837%** | High catch rate of genuine fraud |
| **F1-Score** | **0.8182** | 0.1142 | Superior balance on imbalanced target |
| **Specificity** | **99.967%** | 97.559% | Legitimate transactions approved |
| **PR-AUC** | **0.8418** | 0.7175 | Precision-Recall curve area |
| **ROC-AUC** | 0.9614 | 0.9721 | Receiver operating curve area |
| **True Positives (TP)** | 81 | 90 | Fraudulent transactions stopped |
| **False Positives (FP)** | **19** | **1,388** | **False alarms inconveniencing customers** |
| **True Negatives (TN)** | 56,845 | 55,476 | Legitimate transactions settled |
| **False Negatives (FN)** | 17 | 8 | Undetected fraud transactions |

---

## 🛠️ Common Errors & Troubleshooting

1. **`Java / winutils.exe not found warning` during PySpark run**:
   - *Cause*: On standalone Windows, Apache Hadoop native binaries (`winutils.exe`) are not present by default.
   - *Fix*: This warning is cosmetic and non-fatal. PySpark automatically falls back to built-in Java file systems and completes execution normally.
2. **`ModuleNotFoundError: No module named 'src'`**:
   - *Cause*: Executing from subdirectories without `PYTHONPATH` set.
   - *Fix*: The scripts include automated `sys.path` resolution. Always run commands from the project root (`Credit-Card Fraud`).
3. **`MemoryError` during model training**:
   - *Cause*: System RAM is under 4 GB.
   - *Fix*: Lower `n_estimators` in `config/config.yaml` to 50 or use `data/sample/sample_transactions.csv` for demonstration.

---

## 🎓 Academic Viva Resources
- **Comprehensive Project Report**: See [`reports/project_report.md`](reports/project_report.md) for the 21-section academic report (Abstract, Literature Review, Methodology, Architecture, Results, and References).
- **Viva Preparation Guide**: See [`reports/viva_questions.md`](reports/viva_questions.md) for 36 detailed questions and answers covering ML theory, imbalance handling, Big Data architecture, and confusion matrix defense.
