# ==============================================================================
# Script 01: Dataset Loading and Technical Inspection
# Project: Credit Card Fraud Detection Using Machine Learning and Big Data
# Packages: dplyr, readr
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

cat("==============================================================================\n")
cat("  STEP 1: CREDIT CARD FRAUD TRANSACTION DATASET INSPECTION\n")
cat("==============================================================================\n\n")

# Configurable dataset path
dataset_path <- file.path("data", "raw", "creditcard.csv")
if (!file.exists(dataset_path)) {
  sample_path <- file.path("data", "sample", "sample_transactions.csv")
  if (file.exists(sample_path)) {
    warning(paste("Full raw dataset not found at", dataset_path, "- falling back to", sample_path))
    dataset_path <- sample_path
  } else {
    stop(paste("Neither raw dataset nor sample dataset found. Please check paths."))
  }
}

# 1. File Metadata Inspection
file_size_mb <- file.info(dataset_path)$size / (1024 * 1024)
cat(sprintf("1. File Location : %s\n", dataset_path))
cat(sprintf("   File Size     : %.2f MB\n", file_size_mb))
cat(sprintf("   File Format   : Comma-Separated Values (CSV)\n\n"))

# 2. Ingestion
cat("Loading transaction records into memory...\n")
transactions <- read_csv(dataset_path, show_col_types = FALSE)

num_rows <- nrow(transactions)
num_cols <- ncol(transactions)
cat(sprintf("2. Dimensions    : %s rows, %d columns\n\n", format(num_rows, big.mark = ","), num_cols))

# 3. Column Schema and Data Types
cat("3. Column Schema and Structure:\n")
str(transactions, list.len = 10)
cat("\n")

# 4. Target Variable Identification
candidate_targets <- c("Class", "is_fraud", "fraud", "isFraud", "target")
target_col <- intersect(candidate_targets, colnames(transactions))[1]

if (is.na(target_col)) {
  stop("Error: Target column could not be automatically detected. Please inspect column names.")
}
cat(sprintf("4. Detected Target Variable: '%s'\n", target_col))

# Class Distribution Analysis
class_table <- table(transactions[[target_col]])
fraud_count <- as.numeric(class_table["1"])
legit_count <- as.numeric(class_table["0"])
fraud_pct <- (fraud_count / num_rows) * 100
legit_pct <- (legit_count / num_rows) * 100

cat(sprintf("   - Legitimate Transactions (Class 0): %s (%.4f%%)\n", format(legit_count, big.mark = ","), legit_pct))
cat(sprintf("   - Fraudulent Transactions (Class 1): %s (%.4f%%)\n", format(fraud_count, big.mark = ","), fraud_pct))
cat(sprintf("   - Imbalance Ratio                  : 1 : %.1f\n\n", legit_count / fraud_count))

# 5. Missing Value Audit
missing_per_col <- colSums(is.na(transactions))
total_missing <- sum(missing_per_col)
cat(sprintf("5. Total Missing (NA) Values: %d\n", total_missing))
if (total_missing > 0) {
  cat("   Columns with missing values:\n")
  print(missing_per_col[missing_per_col > 0])
} else {
  cat("   Data Quality Assessment: Zero missing values found across all features.\n\n")
}

# 6. Duplicate Rows Check
duplicate_count <- sum(duplicated(transactions))
cat(sprintf("6. Duplicate Records: %d (%.3f%% of dataset)\n", duplicate_count, (duplicate_count / num_rows) * 100))
cat("   Note: In financial transactions, identical amounts/features can represent retry submissions\n")
cat("   or synthetic SVD collisions. Preprocessing handles these systematically.\n\n")

# 7. Summary of Transaction Time and Amount
cat("7. Summary Statistics for Time (Elapsed Seconds) and Amount:\n")
print(summary(transactions %>% select(Time, Amount)))

cat("\n[Complete] Data inspection verified. Dataset is ready for preprocessing.\n")
