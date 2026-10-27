# CREDIT CARD FRAUD DETECTION — COMPREHENSIVE VIVA VOCE PREPARATION GUIDE

This document contains **36 essential questions and structured answers** designed to help you confidently defend this project in front of college faculty, external examiners, and technical interviewers.

---

### SECTION 1: CORE MACHINE LEARNING & FRAUD DETECTION CONCEPTS

#### Q1. What is credit card fraud detection?
**Answer**: Credit card fraud detection is the automated process of identifying unauthorized, fraudulent financial transactions among millions of legitimate transactions in order to prevent monetary loss and protect cardholders.

#### Q2. Why is machine learning better suited for this problem than traditional rule-based systems?
**Answer**: Traditional rule-based systems rely on static IF-THEN thresholds (e.g., "if amount > $5,000, flag transaction"). Fraudsters quickly learn and circumvent these static limits. Furthermore, static rules generate high false positive rates. Machine learning models analyze multi-dimensional, non-linear relationships across dozens of features simultaneously, adapting to subtle and emerging fraud patterns without requiring manual rule reprogramming.

#### Q3. What is supervised learning?
**Answer**: Supervised learning is a branch of machine learning where an algorithm is trained on a labeled dataset consisting of input features ($X$) and known ground-truth output targets ($y$). The model learns a mathematical mapping function $f(X) \approx y$ that can generalize to predict labels on unseen future inputs.

#### Q4. What is classification?
**Answer**: Classification is a supervised learning task where the target output variable is categorical rather than continuous. In our project, it is **binary classification**, where each transaction is assigned to one of two discrete classes: `0` (Legitimate) or `1` (Fraudulent).

#### Q5. What are features and target variables in this dataset?
**Answer**:
- **Features ($X$)**: The 30 input attributes used to describe each transaction: `Time` (elapsed seconds), `Amount` (transaction value), and `V1` to `V28` (anonymized principal components).
- **Target Variable ($y$)**: The ground-truth label column `Class`, where `0` indicates a legitimate transaction and `1` indicates a fraudulent transaction.

#### Q6. What is class imbalance, and what is the imbalance ratio in our dataset?
**Answer**: Class imbalance occurs when one class significantly outnumbers the other in the training data. In our dataset of 284,807 transactions, only **492 transactions (0.1727%)** are fraudulent, while **284,315 (99.8273%)** are legitimate. This represents an extreme imbalance ratio of approximately **1 : 578**.

#### Q7. Why is Accuracy a misleading evaluation metric for fraud detection?
**Answer**: Accuracy measures total correct predictions over total predictions: $\frac{TP + TN}{\text{Total}}$. In a dataset with 99.83% legitimate transactions, a naive model that predicts "Legitimate" for every single transaction achieves **99.83% accuracy** despite failing to detect a single fraudulent transaction (0% Recall). Therefore, metrics like **Precision, Recall, F1-Score, and PR-AUC** are essential for evaluating imbalanced fraud models.

#### Q8. What are Precision, Recall, and F1-Score?
**Answer**:
- **Precision**: $\frac{TP}{TP + FP}$ — Out of all transactions flagged by the model as fraud, what proportion was actually fraudulent? (High precision means few false alarms).
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ — Out of all actual fraudulent transactions in the dataset, what proportion did the model successfully catch? (High recall means few missed frauds).
- **F1-Score**: $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ — The harmonic mean of precision and recall, providing a balanced single metric when false positives and false negatives carry significant costs.

#### Q9. What is a Confusion Matrix? Explain TP, TN, FP, and FN in the context of fraud detection.
**Answer**: A confusion matrix is a $2 \times 2$ contingency table summarizing the actual versus predicted outcomes:
- **True Positive (TP)**: A fraudulent transaction correctly identified and flagged by the model (Fraud caught).
- **True Negative (TN)**: A legitimate transaction correctly cleared as non-fraudulent (Legitimate cleared).
- **False Positive (FP - Type I Error)**: A legitimate transaction mistakenly flagged as fraud (False alarm, inconveniencing a genuine customer).
- **False Negative (FN - Type II Error)**: A fraudulent transaction missed by the model (Undetected fraud resulting in direct financial loss).

#### Q10. What is Specificity?
**Answer**: Specificity (True Negative Rate) is $\frac{TN}{TN + FP}$. It measures the proportion of actual legitimate transactions that the model correctly identifies as legitimate. In our Random Forest model, specificity is **99.967%**.

#### Q11. What is the difference between ROC-AUC and Precision-Recall AUC (PR-AUC)?
**Answer**:
- **ROC-AUC**: Measures the area under the curve plotting True Positive Rate (Recall) against False Positive Rate ($\frac{FP}{TN+FP}$). Because $TN$ is massive in imbalanced datasets, the False Positive Rate remains tiny even when $FP$ is large, making ROC-AUC overly optimistic.
- **PR-AUC (Average Precision)**: Plots Precision against Recall. It evaluates performance strictly focused on the minority positive class without being inflated by a large number of true negatives, making it the superior benchmark for imbalanced fraud detection.

---

### SECTION 2: DATA PREPROCESSING & METHODOLOGY

#### Q12. Why is data preprocessing necessary?
**Answer**: Real-world data contains variations in scale, missing records, potential duplicates, and anomalies that degrade machine learning algorithms. Preprocessing ensures that input distributions are standardized, outliers do not distort model weights, and the target variable is encoded appropriately.

#### Q13. Why did we use RobustScaler instead of StandardScaler or MinMaxScaler?
**Answer**: `StandardScaler` uses mean and standard deviation, and `MinMaxScaler` relies on minimum and maximum values; both are severely sensitive to extreme outliers. Our `Amount` column contains transactions up to €25,691 while the median is just €22. `RobustScaler` scales data using the **Median** and **Interquartile Range (IQR = $Q_3 - Q_1$)**, ensuring that extreme monetary outliers do not distort the feature representation.

#### Q14. What is data leakage, and how did we prevent it?
**Answer**: Data leakage occurs when information from outside the training dataset (such as test or validation records) is used to create or fit the model. If a scaler or SMOTE oversampler is fitted on the entire dataset prior to splitting, statistics from the test set leak into the training process, producing artificially inflated test scores. We prevented leakage by performing a **stratified split first**, fitting our `RobustScaler` **strictly on the training partition**, and then applying that fitted transformation to the test partition.

#### Q15. Why did we choose Cost-Sensitive Class Weighting over SMOTE?
**Answer**: Synthetic Minority Over-sampling Technique (SMOTE) creates synthetic samples by interpolating between nearest neighbors in feature space. In high-dimensional PCA spaces ($V_1$ to $V_{28}$), synthetic interpolation can generate non-viable combinations that do not reflect genuine fraud tactics. **Cost-Sensitive Learning** (`class_weight='balanced'`) leaves the training data distribution intact but mathematically penalizes minority misclassifications inversely proportional to class frequencies:
$$w_j = \frac{N}{k \cdot n_j}$$
This avoids synthetic artifacts while forcing the decision trees to prioritize fraud detection.

#### Q16. Why did we use Stratified Splitting?
**Answer**: In an 80/20 random split of an imbalanced dataset, a standard unstratified split might randomly distribute all 492 frauds unevenly (or leave too few in the test set). Stratified splitting forces both the training and test sets to maintain the exact same minority class proportion (~0.172%).

---

### SECTION 3: MODEL ARCHITECTURE & RANDOM FOREST

#### Q17. What is Random Forest, and how does it work?
**Answer**: Random Forest is an ensemble machine learning algorithm developed by Leo Breiman that constructs a forest of hundreds of diverse decision trees. It utilizes two core mechanisms:
1. **Bootstrap Aggregation (Bagging)**: Each tree is trained on a random bootstrap sample drawn with replacement from the training data.
2. **Random Subspace Method**: At each node split, only a random subset of features (typically $\sqrt{p}$) is evaluated.
The predictions from all trees are aggregated via majority voting or probability averaging.

#### Q18. Why was Random Forest selected as the primary algorithm?
**Answer**:
1. It handles complex, non-linear interactions across the 28 PCA features without needing manual interaction engineering.
2. It is resistant to overfitting because bagging reduces model variance.
3. It handles tabular data significantly better than unregularized deep neural networks on small-sample minority classes.
4. It provides intrinsic feature importance scores and calibrated probability outputs.

#### Q19. What hyperparameters did you tune in Random Forest?
**Answer**:
- `n_estimators = 100`: The number of decision trees in the ensemble.
- `max_depth = 15`: Constrains tree depth to prevent leaf-level memorization of noise.
- `min_samples_split = 5`: Minimum samples required to split an internal node.
- `min_samples_leaf = 2`: Minimum samples required at each terminal leaf node.
- `class_weight = "balanced"`: Automatically penalizes fraud misclassification.

#### Q20. What is Feature Importance, and which features were most influential?
**Answer**: Feature importance in Random Forest measures the total reduction in **Gini Impurity** brought by splits on a given feature across all trees. In our model, features **$V_{17}, V_{14}, V_{12}, V_{10},$ and $V_{11}$** demonstrated the highest Gini importance, indicating they contain the strongest predictive signals for distinguishing fraudulent transactions.

#### Q21. What is Overfitting, and how did we prevent it?
**Answer**: Overfitting occurs when a model memorizes idiosyncrasies and noise in the training set, achieving near-perfect training accuracy but failing to generalize to new data. We prevented overfitting by:
- Limiting tree depth (`max_depth = 15`).
- Requiring minimum leaf sizes (`min_samples_leaf = 2`).
- Using Random Forest's inherent bagging diversity.
- Validating performance on a completely held-out, untouched test set.

---

### SECTION 4: BIG DATA & DISTRIBUTED COMPUTING (HADOOP & SPARK)

#### Q22. What is Hadoop, and what is HDFS?
**Answer**:
- **Apache Hadoop**: An open-source framework designed for distributed storage and processing of massive datasets across clusters of commodity hardware.
- **HDFS (Hadoop Distributed File System)**: Hadoop's distributed storage layer. It splits large files into blocks (typically 128 MB) and distributes them redundantly across multiple DataNodes managed by a centralized NameNode to provide fault tolerance.

#### Q23. What are the key HDFS commands used in this project?
**Answer**:
- `hdfs dfs -mkdir -p /fraud_detection/input`: Creates a distributed input directory.
- `hdfs dfs -put data/raw/creditcard.csv /fraud_detection/input/`: Uploads the local transaction CSV to HDFS.
- `hdfs dfs -ls /fraud_detection/`: Lists distributed files.
- `hdfs dfs -cat /path`: Displays file contents from HDFS.

#### Q24. What is Apache Spark, and why is it faster than Hadoop MapReduce?
**Answer**: Apache Spark is a unified, in-memory distributed compute engine. Spark is up to 100 times faster than Hadoop MapReduce because it processes data in RAM using **Resilient Distributed Datasets (RDDs)** and directed acyclic graph (DAG) execution, whereas MapReduce writes intermediate calculation states to physical disks after every map and reduce phase.

#### Q25. What is PySpark, and what role did it play in our architecture?
**Answer**: PySpark is the Python API for Apache Spark. In our architecture, PySpark serves as the Big Data analytical pipeline. It reads the raw transaction dataset from distributed storage, performs parallel data cleaning, calculates hourly fraud velocity across 48 elapsed hours, and aggregates monetary amount brackets to feed operational metrics to the dashboard.

#### Q26. Why demonstrate a Big Data pipeline if the dataset is 144 MB?
**Answer**: While 144 MB easily fits in laptop RAM, real banking payment rails (such as Visa or Mastercard) process over 500 million transactions daily, amounting to hundreds of gigabytes per day. Implementing the pipeline in PySpark demonstrates that our data architecture and aggregation logic are horizontally scalable and can run on multi-node production clusters (e.g., AWS EMR or Databricks) without rewriting business logic.

---

### SECTION 5: EXPERIMENTAL RESULTS & VIVA DEFENSE

#### Q27. What are the actual measured results of your Random Forest model?
**Answer** (Based on the actual held-out test set of 56,962 transactions):
- **Accuracy**: 99.937%
- **Precision**: 81.000%
- **Recall**: 82.653%
- **F1-Score**: 0.8182
- **Specificity**: 99.967%
- **ROC-AUC**: 0.9614
- **PR-AUC**: 0.8418
- **Confusion Matrix**: 81 True Positives, 56,845 True Negatives, 19 False Positives, 17 False Negatives.

#### Q28. How does Random Forest compare against the baseline Logistic Regression model?
**Answer**: The baseline Logistic Regression model achieved 97.55% accuracy and caught 90 frauds (91.84% recall), but it generated **1,388 false alarms** (Precision of only 6.09%, F1-Score of 0.1142). In comparison, our Random Forest model caught 81 frauds with only **19 false alarms** (Precision: 81.00%, F1-Score: 0.8182). Random Forest reduced false alarms by **98.6%**, proving that non-linear tree ensembles are essential for keeping operational investigation costs manageable.

#### Q29. How does the decision threshold ($0.50$) impact Precision and Recall?
**Answer**: The decision threshold is the probability cutoff above which a transaction is flagged as fraud ($P(\text{Fraud}) \ge \theta$):
- **Lowering threshold (e.g., $\theta = 0.20$)**: Increases Recall (catches more fraud) but decreases Precision (increases false alarms).
- **Raising threshold (e.g., $\theta = 0.80$)**: Increases Precision (flags only transactions with high certainty) but decreases Recall (misses more fraud).
Our system defaults to $\theta = 0.50$, but exposes an interactive slider in the Streamlit UI so risk officers can calibrate the threshold based on institutional risk tolerance.

#### Q30. How does a single new transaction get scored during live inference?
**Answer**:
1. The user inputs transaction values (`Time`, `Amount`, and $V_1$ to $V_{28}$).
2. The system checks schema validity and feature completeness.
3. The fitted `RobustScaler` scales `Time` and `Amount` using training median and IQR.
4. The scaled vector is fed to `model.predict_proba()`, which computes the fraction of decision trees voting for fraud.
5. The resulting probability is compared against the threshold to assign a risk tier ("Low Risk", "Moderate Risk", "High Risk", or "Critical Risk") and operational verdict.

#### Q31. What are the operational risk tiers defined in the dashboard?
**Answer**:
- **Low Risk ($P < 0.20$)**: Approved for instant automated settlement.
- **Moderate Risk ($0.20 \le P < 0.50$)**: Cleared with ongoing passive behavioral monitoring.
- **High Risk ($0.50 \le P < 0.80$)**: Held for secondary validation (e.g., SMS OTP or push notification).
- **Critical Risk ($P \ge 0.80$)**: Flagged for immediate human investigation and temporary card block.

#### Q32. What are the key limitations of this project?
**Answer**:
1. **Static Dataset**: The data represents a 48-hour snapshot from 2013, lacking modern fraud modalities (e.g., crypto-layering, authorized push payment scams).
2. **PCA Anonymization**: Because features $V_1-V_{28}$ are anonymized principal components, we cannot extract semantic business reasons (like cardholder country or merchant category).
3. **Batch Nature**: The current inference engine runs as a web request rather than an asynchronous event-driven streaming pipeline.

#### Q33. How would you extend this system into a real-time banking fraud prevention system?
**Answer**:
1. Replace batch ingestion with **Apache Kafka** to ingest real-time authorization message streams (ISO 8583 format).
2. Deploy the trained Random Forest model onto a streaming engine (**Apache Spark Structured Streaming** or **Apache Flink**) to evaluate transactions in under 50 milliseconds.
3. Integrate a low-latency distributed key-value store (like **Redis**) to track real-time velocity metrics (e.g., number of transactions on card $X$ in the last 5 minutes).

#### Q34. What improvements would you make with additional time and data?
**Answer**:
1. Implement **Graph Neural Networks (GNNs)** or Graph Databases (Neo4j) to detect fraud rings, shared device fingerprints, and money-mule networks.
2. Incorporate **SHAP (SHapley Additive exPlanations)** values into the Streamlit dashboard to give investigators plain-English explanations for each flagged transaction.
3. Implement adaptive retraining with concept drift detectors (e.g., ADWIN) to combat evolving fraud schemes.

#### Q35. What is the role of the R implementation versus the Python implementation?
**Answer**: The presentation and curriculum require R, RStudio, `dplyr`, `ggplot2`, and `caret` as the primary academic statistical framework. The R scripts (`R/01` to `R/06`) provide a complete, standalone R implementation. Python was utilized to build the production Big Data pipeline (PySpark) and the interactive web interface (Streamlit) for enterprise deployment.

#### Q36. If an examiner asks: "Can you prove that your model didn't just memorize the 492 fraud cases?", what is your answer?
**Answer**: Yes, through two strict safeguards:
1. All reported metrics were evaluated on **unseen, held-out test data (56,962 transactions)** that the model never encountered during training.
2. The trees were constrained with `max_depth = 15` and `min_samples_leaf = 2`, preventing individual trees from growing degenerate leaves that memorize individual fraud instances. The model achieved **81% precision and 82.65% recall on the unseen test set**, confirming genuine pattern generalization.
