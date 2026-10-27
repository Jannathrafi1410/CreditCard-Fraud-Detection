# ==============================================================================
# Credit Card Fraud Detection - R Package Installation Script
# Technology Stack: R, RStudio, dplyr, ggplot2, caret, randomForest, PRROC, pROC
# ==============================================================================

cat("Checking and installing required R packages for Credit Card Fraud Detection...\n")

required_packages <- c(
  "dplyr",         # Data manipulation and pipeline verbs
  "ggplot2",       # Publication-quality data visualization
  "caret",         # Machine learning workflow, training, and evaluation
  "randomForest",  # Breiman's Random Forest classification algorithm
  "pROC",          # ROC curve computation and AUC calculation
  "PRROC",         # Precision-Recall curve computation for imbalanced data
  "e1071",         # Support vector / statistical helper for caret
  "corrplot",      # Correlation matrix visualizations
  "readr",         # Fast CSV reading and parsing
  "scales"         # Formatting helpers for ggplot2 scales
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(paste0("Installing missing package: '", pkg, "' from CRAN...\n"))
    install.packages(pkg, repos = "https://cloud.r-project.org/", dependencies = TRUE)
  } else {
    cat(paste0("Package '", pkg, "' is already installed.\n"))
  }
}

invisible(sapply(required_packages, install_if_missing))

cat("\nAll required R packages are installed and ready for analysis!\n")
