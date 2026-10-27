# ==============================================================================
# Script 02: Data Cleaning, Robust Scaling, and Stratified Splitting
# Project: Credit Card Fraud Detection Using Machine Learning and Big Data
# Packages: dplyr, caret, readr
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
  library(readr)
})

cat("==============================================================================\n")
cat("  STEP 2: DATA PREPROCESSING AND LEAK-FREE SPLITTING PIPELINE\n")
cat("==============================================================================\n\n")

# Set seed for exact academic reproducibility
set.seed(42)

# Load data
dataset_path <- if (file.exists("data/raw/creditcard.csv")) "data/raw/creditcard.csv" else "data/sample/sample_transactions.csv"
transactions <- read_csv(dataset_path, show_col_types = FALSE)

cat(sprintf("Loaded %s records for preprocessing.\n", format(nrow(transactions), big.mark = ",")))

# 1. Format Target Variable for Caret Workflow
# Convert binary 0/1 to descriptive factor levels ("Legit" vs "Fraud")
transactions$Class <- factor(
  ifelse(transactions$Class == 1, "Fraud", "Legit"),
  levels = c("Legit", "Fraud")
)

# 2. Stratified Train-Test Split (80% Train, 20% Test)
# Stratification ensures minority fraud proportion is strictly preserved
cat("Performing stratified train-test partition (80% Train, 20% Held-Out Test)...\n")
train_indices <- createDataPartition(transactions$Class, p = 0.80, list = FALSE)

train_data <- transactions[train_indices, ]
test_data  <- transactions[-train_indices, ]

cat(sprintf("Training Set: %s rows (Fraud: %d, %.4f%%)\n",
            format(nrow(train_data), big.mark = ","),
            sum(train_data$Class == "Fraud"),
            mean(train_data$Class == "Fraud") * 100))

cat(sprintf("Testing Set : %s rows (Fraud: %d, %.4f%%)\n",
            format(nrow(test_data), big.mark = ","),
            sum(test_data$Class == "Fraud"),
            mean(test_data$Class == "Fraud") * 100))

# 3. Robust Scaling for Time and Amount
# PCA features V1-V28 are already standardized. Time and Amount have extreme outliers.
# Fit median and IQR strictly on training data to prevent data leakage.
cat("\nFitting Robust Scaler on training partition (Time and Amount)...\n")

time_median   <- median(train_data$Time)
time_iqr      <- IQR(train_data$Time)
amount_median <- median(train_data$Amount)
amount_iqr    <- IQR(train_data$Amount)

# Apply transformation
train_data$Time_Scaled   <- (train_data$Time - time_median) / time_iqr
train_data$Amount_Scaled <- (train_data$Amount - amount_median) / amount_iqr

test_data$Time_Scaled   <- (test_data$Time - time_median) / time_iqr
test_data$Amount_Scaled <- (test_data$Amount - amount_median) / amount_iqr

# Drop unscaled original Time and Amount columns
train_prepared <- train_data %>% select(-Time, -Amount)
test_prepared  <- test_data %>% select(-Time, -Amount)

# 4. Save Processed Datasets and Scaler Parameters
dir.create("data/processed", showWarnings = FALSE, recursive = TRUE)
dir.create("models", showWarnings = FALSE, recursive = TRUE)

scaler_params <- list(
  time_median = time_median,
  time_iqr = time_iqr,
  amount_median = amount_median,
  amount_iqr = amount_iqr
)

save(scaler_params, file = "models/r_scaler_params.RData")
save(train_prepared, file = "data/processed/train_r.RData")
save(test_prepared, file = "data/processed/test_r.RData")

cat("[Complete] Preprocessed datasets and scaler parameters successfully saved to 'data/processed/' and 'models/'.\n")
