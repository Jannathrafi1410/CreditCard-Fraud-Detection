# ==============================================================================
# Script 06: Scoring and Inference on New Transaction Records
# Project: Credit Card Fraud Detection Using Machine Learning and Big Data
# Packages: randomForest, dplyr, readr
# ==============================================================================

suppressPackageStartupMessages({
  library(randomForest)
  library(dplyr)
  library(readr)
})

cat("==============================================================================\n")
cat("  STEP 6: TRANSACTION INFERENCE AND SCORING (R PIPELINE)\n")
cat("==============================================================================\n\n")

# Load model and scaler artifacts
if (!file.exists("models/r_random_forest.RData") || !file.exists("models/r_scaler_params.RData")) {
  stop("Model artifacts not found. Run scripts 02 and 04 first.")
}

load("models/r_random_forest.RData")
load("models/r_scaler_params.RData")

# Input file for batch scoring
batch_file <- "data/sample/test_batch_transactions.csv"
if (!file.exists(batch_file)) {
  stop(paste("Batch transaction file not found at", batch_file))
}

cat(sprintf("Loading incoming transaction batch from: %s\n", batch_file))
new_transactions <- read_csv(batch_file, show_col_types = FALSE)

# 1. Feature Preprocessing using Saved Scaler Parameters
cat("Applying fitted RobustScaler transformations to Time and Amount...\n")
prepared_batch <- new_transactions %>%
  mutate(
    Time_Scaled   = (Time - scaler_params$time_median) / scaler_params$time_iqr,
    Amount_Scaled = (Amount - scaler_params$amount_median) / scaler_params$amount_iqr
  ) %>%
  select(-Time, -Amount)

# 2. Probability Scoring
cat("Evaluating transaction risks with trained Random Forest ensemble...\n")
pred_probs <- predict(rf_model, newdata = prepared_batch, type = "prob")
fraud_probabilities <- pred_probs[, "Fraud"]

# 3. Decision Rule & Operational Risk Tiers
threshold <- 0.50
new_transactions$Fraud_Probability <- round(fraud_probabilities, 4)
new_transactions$Predicted_Class   <- ifelse(fraud_probabilities >= threshold, 1, 0)
new_transactions$Verdict           <- ifelse(fraud_probabilities >= threshold, "Flagged for Investigation", "Approved Legitimate")

new_transactions$Risk_Tier <- case_when(
  fraud_probabilities >= 0.80 ~ "Critical Risk",
  fraud_probabilities >= threshold ~ "High Risk",
  fraud_probabilities >= 0.20 ~ "Moderate Risk",
  TRUE ~ "Low Risk"
)

# 4. Summary of Scored Batch
cat("\n--- Scoring Summary for Incoming Batch ---\n")
cat(sprintf("Total Transactions Scored : %d\n", nrow(new_transactions)))
cat(sprintf("Flagged for Investigation : %d (%.2f%%)\n",
            sum(new_transactions$Predicted_Class == 1),
            mean(new_transactions$Predicted_Class == 1) * 100))
cat(sprintf("Approved Legitimate       : %d (%.2f%%)\n\n",
            sum(new_transactions$Predicted_Class == 0),
            mean(new_transactions$Predicted_Class == 0) * 100))

cat("Sample of High-Risk / Flagged Transactions:\n")
flagged <- new_transactions %>% filter(Predicted_Class == 1) %>% select(Fraud_Probability, Risk_Tier, Verdict, Amount) %>% head(5)
print(flagged)

# 5. Export Scored Batch
output_path <- "data/sample/scored_batch_r_output.csv"
write_csv(new_transactions, output_path)
cat(sprintf("\n[Complete] Scored transaction records exported to '%s'.\n", output_path))
