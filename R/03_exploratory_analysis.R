# ==============================================================================
# Script 03: Exploratory Data Analysis (EDA) and Visualizations
# Project: Credit Card Fraud Detection Using Machine Learning and Big Data
# Packages: ggplot2, dplyr, readr, corrplot, scales
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(scales)
})

cat("==============================================================================\n")
cat("  STEP 3: EXPLORATORY DATA ANALYSIS (EDA) VIA GGPLOT2\n")
cat("==============================================================================\n\n")

dir.create("reports/figures", showWarnings = FALSE, recursive = TRUE)

dataset_path <- if (file.exists("data/raw/creditcard.csv")) "data/raw/creditcard.csv" else "data/sample/sample_transactions.csv"
transactions <- read_csv(dataset_path, show_col_types = FALSE)

# 1. Class Distribution Plot (Log Scale)
cat("Generating Plot 1: Class Distribution Barplot...\n")
class_summary <- transactions %>%
  group_by(Class) %>%
  summarise(Count = n()) %>%
  mutate(
    Label = ifelse(Class == 1, "Fraudulent (1)", "Legitimate (0)"),
    Percentage = (Count / sum(Count)) * 100
  )

p1 <- ggplot(class_summary, aes(x = Label, y = Count, fill = Label)) +
  geom_bar(stat = "identity", width = 0.5, show.legend = FALSE) +
  scale_y_log10(labels = comma) +
  scale_fill_manual(values = c("Legitimate (0)" = "#2563eb", "Fraudulent (1)" = "#dc2626")) +
  geom_text(aes(label = sprintf("%s\n(%.3f%%)", comma(Count), Percentage)), vjust = -0.3, fontface = "bold", size = 4) +
  labs(
    title = "Class Distribution in Credit Card Transactions (Log Scale)",
    subtitle = "Severe Class Imbalance: Fraud represents ~0.172% of total transaction volume",
    x = "Transaction Category",
    y = "Transaction Count (Log10 Scale)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("reports/figures/r_class_distribution.png", p1, width = 7, height = 5, dpi = 300)

# 2. Transaction Amount Distribution
cat("Generating Plot 2: Transaction Amount Distribution...\n")
p2 <- ggplot(transactions, aes(x = Amount)) +
  geom_histogram(bins = 60, fill = "#3b82f6", color = "white", alpha = 0.8) +
  scale_y_log10(labels = comma) +
  labs(
    title = "Transaction Amount Frequency Distribution",
    subtitle = "Heavily skewed distribution with extreme high-value outliers up to €25,691",
    x = "Transaction Amount (€)",
    y = "Frequency (Log10 Scale)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("reports/figures/r_amount_distribution.png", p2, width = 8, height = 5, dpi = 300)

# 3. Transaction Amount Comparison by Class (Boxplot)
cat("Generating Plot 3: Transaction Amount Comparison by Class...\n")
p3 <- ggplot(transactions, aes(x = factor(Class, labels = c("Legitimate", "Fraudulent")), y = Amount, fill = factor(Class))) +
  geom_boxplot(outlier.shape = NA, show.legend = FALSE, width = 0.4) +
  coord_cartesian(ylim = c(0, 250)) +
  scale_fill_manual(values = c("#93c5fd", "#fca5a5")) +
  labs(
    title = "Comparison of Transaction Amounts by Class (Truncated at €250)",
    subtitle = "Fraudulent transactions often cluster around specific moderate values, avoiding high thresholds",
    x = "Transaction Class",
    y = "Transaction Amount (€)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("reports/figures/r_amount_by_class.png", p3, width = 7, height = 5, dpi = 300)

# 4. Temporal Density of Transactions
cat("Generating Plot 4: Temporal Density by Class...\n")
p4 <- ggplot(transactions, aes(x = Time, color = factor(Class, labels = c("Legitimate", "Fraudulent")), fill = factor(Class, labels = c("Legitimate", "Fraudulent")))) +
  geom_density(alpha = 0.3, linewidth = 1.0) +
  scale_color_manual(values = c("Legitimate" = "#2563eb", "Fraudulent" = "#dc2626")) +
  scale_fill_manual(values = c("Legitimate" = "#2563eb", "Fraudulent" = "#dc2626")) +
  labs(
    title = "Kernel Density Estimation Across Elapsed Time",
    subtitle = "Elapsed seconds over 48 hours; legitimate transactions follow diurnal patterns",
    x = "Time Elapsed (seconds from first transaction)",
    y = "Density",
    color = "Class", fill = "Class"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), legend.position = "top")

ggsave("reports/figures/r_time_density.png", p4, width = 9, height = 5, dpi = 300)

cat("[Complete] All ggplot2 exploratory figures generated and saved to 'reports/figures/'.\n")
