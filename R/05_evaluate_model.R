# ==============================================================================
# Script 05: Model Evaluation on Unseen Held-Out Test Set
# Project: Credit Card Fraud Detection Using Machine Learning and Big Data
# Packages: caret, randomForest, pROC, PRROC, ggplot2, dplyr
# ==============================================================================

suppressPackageStartupMessages({
  library(caret)
  library(randomForest)
  library(pROC)
  library(PRROC)
  library(ggplot2)
  library(dplyr)
})

cat("==============================================================================\n")
cat("  STEP 5: EVALUATION ON HELD-OUT TEST PARTITION (R PIPELINE)\n")
cat("==============================================================================\n\n")

# Load model and test dataset
if (!file.exists("models/r_random_forest.RData") || !file.exists("data/processed/test_r.RData")) {
  stop("Model or test dataset missing. Please run scripts 02 and 04 first.")
}

load("models/r_random_forest.RData")
load("data/processed/test_r.RData")

cat(sprintf("Loaded test partition with %s unseen transactions.\n", format(nrow(test_prepared), big.mark = ",")))

# 1. Generate Probability Predictions on Test Set
cat("Computing probabilistic risk scores for test transactions...\n")
pred_prob <- predict(rf_model, newdata = test_prepared, type = "prob")
fraud_probabilities <- pred_prob[, "Fraud"]

# 2. Apply Decision Threshold (0.50 default)
decision_threshold <- 0.50
pred_classes <- factor(ifelse(fraud_probabilities >= decision_threshold, "Fraud", "Legit"), levels = c("Legit", "Fraud"))

# 3. Compute Confusion Matrix via Caret
cm <- confusionMatrix(pred_classes, test_prepared$Class, positive = "Fraud")
cat("\nConfusion Matrix & Core Diagnostic Statistics:\n")
print(cm)

# Extract core metrics
tp <- cm$table["Fraud", "Fraud"]
tn <- cm$table["Legit", "Legit"]
fp <- cm$table["Fraud", "Legit"]
fn <- cm$table["Legit", "Fraud"]

accuracy    <- (tp + tn) / (tp + tn + fp + fn)
precision   <- tp / (tp + fp)
recall      <- tp / (tp + fn) # Sensitivity
specificity <- tn / (tn + fp)
f1_score    <- 2 * (precision * recall) / (precision + recall)

# 4. Compute ROC-AUC via pROC
binary_actual <- ifelse(test_prepared$Class == "Fraud", 1, 0)
roc_obj <- roc(binary_actual, fraud_probabilities, quiet = TRUE)
roc_auc <- as.numeric(auc(roc_obj))

# 5. Compute PR-AUC via PRROC
pr_obj <- pr.curve(scores.class0 = fraud_probabilities[binary_actual == 1],
                   scores.class1 = fraud_probabilities[binary_actual == 0],
                   curve = TRUE)
pr_auc <- pr_obj$auc.integral

cat("\n" + paste(rep("=", 60), collapse = "") + "\n")
cat("  FINAL MEASURED PERFORMANCE SUMMARY (R RANDOM FOREST)\n")
cat(paste(rep("=", 60), collapse = "") + "\n")
cat(sprintf("  Accuracy           : %.4f (%.2f%%)\n", accuracy, accuracy * 100))
cat(sprintf("  Precision          : %.4f (%.2f%%)\n", precision, precision * 100))
cat(sprintf("  Recall (Sensitivity: %.4f (%.2f%%)\n", recall, recall * 100))
cat(sprintf("  F1-Score           : %.4f\n", f1_score))
cat(sprintf("  Specificity        : %.4f (%.2f%%)\n", specificity, specificity * 100))
cat(sprintf("  ROC-AUC            : %.4f\n", roc_auc))
cat(sprintf("  PR-AUC             : %.4f\n\n", pr_auc))
cat("Operational Transaction Outcomes:\n")
cat(sprintf("  - True Positives  (TP) : %d (Fraudulent transactions correctly caught)\n", tp))
cat(sprintf("  - True Negatives  (TN) : %s (Legitimate transactions correctly approved)\n", format(tn, big.mark = ",")))
cat(sprintf("  - False Positives (FP) : %d (False alarms - legitimate transactions held)\n", fp))
cat(sprintf("  - False Negatives (FN) : %d (Missed fraud transactions)\n", fn))
cat(paste(rep("=", 60), collapse = "") + "\n\n")

# 6. Save Metrics to CSV
dir.create("reports/metrics", showWarnings = FALSE, recursive = TRUE)
metrics_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1_Score", "Specificity", "ROC_AUC", "PR_AUC", "TP", "TN", "FP", "FN"),
  Value = c(accuracy, precision, recall, f1_score, specificity, roc_auc, pr_auc, tp, tn, fp, fn)
)
write.csv(metrics_df, "reports/metrics/r_evaluation_metrics.csv", row.names = FALSE)
cat("Exported evaluation metrics to 'reports/metrics/r_evaluation_metrics.csv'.\n")

# 7. Render ROC Curve Plot
png("reports/figures/r_roc_curve.png", width = 2100, height = 1800, res = 300)
plot(roc_obj, col = "#2563eb", lwd = 2.5, main = sprintf("ROC Curve (AUC = %.4f)", roc_auc))
abline(a = 0, b = 1, lty = 2, col = "red")
grid(col = "lightgray")
dev.off()

cat("[Complete] Evaluation pipeline concluded. Diagnostic plots saved.\n")
