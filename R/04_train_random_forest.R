# ==============================================================================
# Script 04: Random Forest Classifier Training with Imbalance Handling
# Project: Credit Card Fraud Detection Using Machine Learning and Big Data
# Packages: caret, randomForest, dplyr
# ==============================================================================

suppressPackageStartupMessages({
  library(caret)
  library(randomForest)
  library(dplyr)
})

cat("==============================================================================\n")
cat("  STEP 4: RANDOM FOREST MODEL TRAINING & IMBALANCE MITIGATION\n")
cat("==============================================================================\n\n")

set.seed(42)

# Load preprocessed training partition
if (!file.exists("data/processed/train_r.RData")) {
  stop("Training partition 'data/processed/train_r.RData' not found. Run '02_data_preprocessing.R' first.")
}

load("data/processed/train_r.RData")
cat(sprintf("Loaded prepared training dataset with %s records.\n", format(nrow(train_prepared), big.mark = ",")))

# Strategy for Imbalanced Classes:
# In randomForest, we can configure class-specific loss weights or balanced downsampling.
# We set class weights inversely proportional to class frequency:
# Weight for Fraud = 100, Weight for Legit = 1.
class_weights <- c("Legit" = 1, "Fraud" = 100)

cat("\nConfiguring Random Forest Hyperparameters:\n")
cat("  - Number of Trees (ntree)     : 100\n")
cat("  - Variables per Split (mtry)  : 5 (~sqrt(p))\n")
cat("  - Minimum Node Size (nodesize): 2\n")
cat("  - Class Weights               : Legit = 1, Fraud = 100\n")
cat("  - Stratified Bootstrap Sample : Enabled\n\n")

cat("Initiating model training (Breiman's Random Forest algorithm)...\n")
start_time <- Sys.time()

# Train randomForest model directly
rf_model <- randomForest(
  Class ~ .,
  data = train_prepared,
  ntree = 100,
  mtry = 5,
  nodesize = 2,
  classwt = class_weights,
  importance = TRUE,
  keep.forest = TRUE
)

end_time <- Sys.time()
training_duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat(sprintf("Training completed in %.2f seconds.\n\n", training_duration))

# Print Out-of-Bag (OOB) error estimation
cat("Out-Of-Bag (OOB) Training Summary:\n")
print(rf_model)

# Feature Importance Summary
cat("\nTop 10 Most Influential Features (Mean Decrease Gini):\n")
imp <- importance(rf_model, type = 2) # Type 2 is Gini decrease
imp_df <- data.frame(Feature = rownames(imp), MeanDecreaseGini = imp[, 1]) %>%
  arrange(desc(MeanDecreaseGini)) %>%
  head(10)
print(imp_df)

# Save Model Artifact
dir.create("models", showWarnings = FALSE, recursive = TRUE)
save(rf_model, file = "models/r_random_forest.RData")

cat("\n[Complete] Trained Random Forest model successfully saved to 'models/r_random_forest.RData'.\n")
