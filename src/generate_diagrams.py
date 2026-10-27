"""
Generates academic-quality system architecture and data flow diagrams (DFDs)
for the Credit Card Fraud Detection project using Matplotlib.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure diagrams directory exists
os.makedirs("diagrams", exist_ok=True)

# Common styling palette
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#2ca02c"
ACCENT_COLOR = "#ff7f0e"
BG_COLOR = "#f8f9fa"
BOX_COLOR = "#e9ecef"
BORDER_COLOR = "#343a40"
TEXT_COLOR = "#212529"
SPARK_COLOR = "#e34a33"


def draw_box(ax, x, y, width, height, text, subtext="", color="#e3f2fd", border="#1976d2", text_color="#0d47a1", fontsize=10):
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.2,rounding_size=0.15",
        edgecolor=border, facecolor=color, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    if subtext:
        ax.text(center_x, center_y + 0.12, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
        ax.text(center_x, center_y - 0.15, subtext, ha="center", va="center", fontsize=fontsize-2.5, color="#495057", style="italic", zorder=4)
    else:
        ax.text(center_x, center_y, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, label="", color="#495057", rad=0.0):
    connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", color=color, lw=1.8, connectionstyle=connectionstyle),
        zorder=2
    )
    if label:
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0 + 0.08
        ax.text(mid_x, mid_y, label, ha="center", va="center", fontsize=8, color="#212529",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8), zorder=5)


def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    ax.text(7, 8.5, "Credit Card Fraud Detection System Architecture", ha="center", va="center", fontsize=16, fontweight="bold", color="#1a252c")
    ax.text(7, 8.1, "Hybrid Machine Learning & Big Data Analytics Architecture", ha="center", va="center", fontsize=10, color="#6c757d")

    # Group Boundaries
    # Core ML Pipeline
    ml_group = patches.FancyBboxPatch((0.5, 3.6), 13.0, 4.0, boxstyle="round,pad=0.3", edgecolor="#90caf9", facecolor="#f3f8fd", linestyle="--", linewidth=1.2, zorder=1)
    ax.add_patch(ml_group)
    ax.text(0.8, 7.3, "Primary ML & Interactive Analytics Pipeline (R / Python Caret & Scikit-Learn)", fontsize=9, fontweight="bold", color="#1565c0")

    # Big Data Pipeline
    spark_group = patches.FancyBboxPatch((0.5, 0.4), 13.0, 2.7, boxstyle="round,pad=0.3", edgecolor="#ffab91", facecolor="#fff8f6", linestyle="--", linewidth=1.2, zorder=1)
    ax.add_patch(spark_group)
    ax.text(0.8, 2.8, "Distributed Big Data Processing Pipeline (Apache Hadoop HDFS & Apache Spark / PySpark)", fontsize=9, fontweight="bold", color="#d84315")

    # ML Pipeline Components
    draw_box(ax, 0.8, 5.0, 1.8, 1.2, "Raw Transactions\nDataset", "creditcard.csv\n(284k records)", color="#e0f2fe", border="#0284c7")
    draw_box(ax, 3.2, 5.0, 1.8, 1.2, "Data Ingestion &\nPreprocessing", "RobustScaler & Imbalance\nStratified 80/20", color="#e0f2fe", border="#0284c7")
    draw_box(ax, 5.6, 5.7, 1.8, 1.0, "Exploratory Data\nAnalysis (EDA)", "ggplot2 / Seaborn\nVisualizations", color="#e8f5e9", border="#2e7d32", text_color="#1b5e20")
    draw_box(ax, 5.6, 4.3, 1.8, 1.0, "Random Forest\nModel Training", "Balanced Weighting\nn_trees=100", color="#fef3c7", border="#d97706", text_color="#92400e")
    draw_box(ax, 8.0, 4.3, 1.8, 1.0, "Model Evaluation &\nValidation", "ROC-AUC, PR-AUC\nConfusion Matrix", color="#fef3c7", border="#d97706", text_color="#92400e")
    draw_box(ax, 8.0, 5.7, 1.8, 1.0, "Model & Preprocessor\nRegistry", "Joblib / RData\nTrained Artifacts", color="#ede9fe", border="#7c3aed", text_color="#5b21b6")
    draw_box(ax, 10.5, 4.8, 2.2, 1.5, "Streamlit Web\nDashboard", "Interactive UI, Single\n& Batch Predictions", color="#fce7f3", border="#db2777", text_color="#9d174d", fontsize=11)

    # ML Arrows
    draw_arrow(ax, 2.6, 5.6, 3.2, 5.6)
    draw_arrow(ax, 5.0, 5.6, 5.6, 6.2)
    draw_arrow(ax, 5.0, 5.6, 5.6, 4.8)
    draw_arrow(ax, 7.4, 4.8, 8.0, 4.8)
    draw_arrow(ax, 8.9, 5.3, 8.9, 5.7, label="Validated")
    draw_arrow(ax, 9.8, 6.2, 10.5, 5.7, label="Load Models")
    draw_arrow(ax, 7.4, 6.2, 10.5, 6.0, label="EDA Reports", rad=-0.1)

    # Spark Pipeline Components
    draw_box(ax, 1.5, 1.0, 2.0, 1.2, "Hadoop HDFS Ingestion", "hdfs dfs -put\nDistributed Storage", color="#ffedd5", border="#ea580c", text_color="#9a3412")
    draw_box(ax, 4.5, 1.0, 2.4, 1.2, "Apache Spark / PySpark", "Distributed Aggregation\nHourly & Amount Profiling", color="#ffedd5", border="#ea580c", text_color="#9a3412")
    draw_box(ax, 8.0, 1.0, 2.2, 1.2, "Aggregated Big Data\nAnalytics", "HDFS Output Parquet\nDistributed Summaries", color="#ffedd5", border="#ea580c", text_color="#9a3412")

    # Big Data Connections
    draw_arrow(ax, 1.7, 5.0, 2.5, 2.2, label="Raw Data Upload", rad=0.2)
    draw_arrow(ax, 3.5, 1.6, 4.5, 1.6, label="Spark RDD/DF")
    draw_arrow(ax, 6.9, 1.6, 8.0, 1.6, label="Action/Transform")
    draw_arrow(ax, 9.1, 2.2, 11.2, 4.8, label="Operational Feeds", rad=-0.2)

    plt.tight_layout()
    plt.savefig("diagrams/system_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved diagrams/system_architecture.png")


def generate_dfd_level_0():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(5, 5.6, "Data Flow Diagram — Level 0 (Context Level)", ha="center", va="center", fontsize=14, fontweight="bold", color="#1a252c")
    ax.text(5, 5.25, "System Context Boundary for Credit Card Fraud Detection", ha="center", va="center", fontsize=9, color="#6c757d")

    # External Entities (Rectangles)
    draw_box(ax, 0.5, 2.2, 2.0, 1.4, "Bank Transaction\nData Source", "Real/Historical\nTransaction Stream", color="#e2e8f0", border="#475569", text_color="#1e293b")
    draw_box(ax, 7.5, 3.2, 2.0, 1.4, "Fraud Analyst /\nCompliance Officer", "Human Investigator\n& Faculty Reviewer", color="#e2e8f0", border="#475569", text_color="#1e293b")
    draw_box(ax, 7.5, 0.8, 2.0, 1.4, "End User /\nClient Application", "Cardholder &\nOperational APIs", color="#e2e8f0", border="#475569", text_color="#1e293b")

    # Central System (Circle / Oval / Round Box)
    rect = patches.FancyBboxPatch(
        (3.8, 1.8), 2.4, 2.2,
        boxstyle="circle,pad=0.2",
        edgecolor="#0284c7", facecolor="#e0f2fe", linewidth=2.0, zorder=3
    )
    ax.add_patch(rect)
    ax.text(5.0, 3.2, "0.0", ha="center", va="center", fontsize=12, fontweight="bold", color="#0369a1", zorder=4)
    ax.text(5.0, 2.8, "Credit Card\nFraud Detection\nSystem", ha="center", va="center", fontsize=11, fontweight="bold", color="#0c4a6e", zorder=4)
    ax.text(5.0, 2.2, "(ML + Big Data)", ha="center", va="center", fontsize=8, color="#0369a1", style="italic", zorder=4)

    # Data Flows (Arrows)
    draw_arrow(ax, 2.5, 3.0, 3.8, 3.0, label="Transaction Records (CSV/Batch)")
    draw_arrow(ax, 6.2, 3.4, 7.5, 3.8, label="Fraud Alerts & Metrics")
    draw_arrow(ax, 6.2, 2.4, 7.5, 1.6, label="Legitimate / Suspicious Verdict")
    draw_arrow(ax, 7.5, 3.4, 6.2, 2.8, label="Decision Threshold Config", rad=0.2)

    plt.tight_layout()
    plt.savefig("diagrams/dfd_level_0.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved diagrams/dfd_level_0.png")


def generate_dfd_level_1():
    fig, ax = plt.subplots(figsize=(13, 8), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6.5, 7.6, "Data Flow Diagram — Level 1 (Functional Decomposition)", ha="center", va="center", fontsize=14, fontweight="bold", color="#1a252c")
    ax.text(6.5, 7.25, "Major Subsystems, Data Stores, and Transformations", ha="center", va="center", fontsize=9, color="#6c757d")

    # Processes
    draw_box(ax, 0.5, 4.2, 2.0, 1.3, "1.0 Data Ingestion\n& Validation", "Schema & Null Check", color="#e0f2fe", border="#0284c7")
    draw_box(ax, 3.2, 4.2, 2.0, 1.3, "2.0 Data Preprocessing\n& Scaling", "RobustScaler & Split", color="#e0f2fe", border="#0284c7")
    draw_box(ax, 6.0, 5.5, 2.0, 1.3, "3.0 Exploratory Data\nAnalysis (EDA)", "ggplot2 / Seaborn", color="#e8f5e9", border="#2e7d32", text_color="#1b5e20")
    draw_box(ax, 6.0, 3.0, 2.0, 1.3, "4.0 Model Training &\nTuning", "Random Forest Classifier", color="#fef3c7", border="#d97706", text_color="#92400e")
    draw_box(ax, 8.8, 3.0, 2.0, 1.3, "5.0 Model Evaluation\n& Benchmarking", "ROC-AUC, Confusion Mat", color="#fef3c7", border="#d97706", text_color="#92400e")
    draw_box(ax, 10.5, 4.8, 2.0, 1.3, "6.0 Inference &\nThresholding", "Real-time & Batch Scoring", color="#fce7f3", border="#db2777", text_color="#9d174d")
    draw_box(ax, 3.2, 1.0, 2.2, 1.3, "7.0 Big Data Storage\n& PySpark Processing", "HDFS & Distributed DF", color="#ffedd5", border="#ea580c", text_color="#9a3412")

    # Data Stores (Parallel lines)
    def draw_datastore(ax, x, y, w, h, name, id_str):
        ax.plot([x, x+w], [y, y], color="#475569", lw=2)
        ax.plot([x, x+w], [y+h, y+h], color="#475569", lw=2)
        rect = patches.Rectangle((x, y), w, h, facecolor="#f1f5f9", edgecolor="none", zorder=2)
        ax.add_patch(rect)
        ax.text(x+0.3, y+h/2, id_str, ha="center", va="center", fontsize=8, fontweight="bold", color="#334155", zorder=3)
        ax.text(x+w/2+0.2, y+h/2, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1e293b", zorder=3)

    draw_datastore(ax, 3.2, 6.3, 2.0, 0.6, "Raw CSV Dataset", "D1")
    draw_datastore(ax, 6.0, 1.5, 2.0, 0.6, "Preprocessed Parquet", "D2")
    draw_datastore(ax, 8.8, 5.5, 2.0, 0.6, "Trained Model Artifacts", "D3")

    # Arrows
    draw_arrow(ax, 3.2, 6.5, 1.5, 5.5, label="Raw Records")
    draw_arrow(ax, 2.5, 4.85, 3.2, 4.85, label="Validated Data")
    draw_arrow(ax, 5.2, 4.85, 6.0, 5.8, label="Cleaned Data")
    draw_arrow(ax, 5.2, 4.85, 6.0, 3.8, label="Train/Test Splits")
    draw_arrow(ax, 7.0, 3.0, 7.0, 2.1, label="Processed")
    draw_arrow(ax, 8.0, 3.65, 8.8, 3.65, label="Trained Trees")
    draw_arrow(ax, 9.8, 4.3, 9.8, 5.5, label="Model Registry")
    draw_arrow(ax, 9.8, 5.5, 10.5, 5.5, label="Joblib Artifact")
    draw_arrow(ax, 2.0, 4.2, 3.2, 1.8, label="HDFS Ingest", rad=-0.1)
    draw_arrow(ax, 5.4, 1.6, 6.0, 1.6, label="Spark Output")

    plt.tight_layout()
    plt.savefig("diagrams/dfd_level_1.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved diagrams/dfd_level_1.png")


def generate_dfd_level_2():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6, 6.6, "Data Flow Diagram — Level 2 (Inference & Decision Subprocess)", ha="center", va="center", fontsize=14, fontweight="bold", color="#1a252c")
    ax.text(6, 6.25, "Detailed Expansion of Preprocessing, Inference, Thresholding, and Alert Generation", ha="center", va="center", fontsize=9, color="#6c757d")

    # Detailed Sub-processes
    draw_box(ax, 0.6, 3.5, 1.8, 1.3, "2.1 Schema &\nType Validation", "Check 30 features", color="#e0f2fe", border="#0284c7")
    draw_box(ax, 2.8, 3.5, 1.8, 1.3, "2.2 Feature Scaling\n(RobustScaler)", "Scale Amount & Time", color="#e0f2fe", border="#0284c7")
    draw_box(ax, 5.0, 3.5, 2.0, 1.3, "2.3 Ensemble Model\nInference", "Random Forest Predict\nProbability P(Fraud)", color="#fef3c7", border="#d97706", text_color="#92400e")
    draw_box(ax, 7.5, 3.5, 2.0, 1.3, "2.4 Risk Scoring &\nThreshold Check", "Compare P >= 0.50\n(Adjustable Cutoff)", color="#ede9fe", border="#7c3aed", text_color="#5b21b6")
    draw_box(ax, 10.0, 4.5, 1.6, 1.1, "2.5 Alert Trigger", "Flag for Review\n(Suspicious)", color="#fee2e2", border="#ef4444", text_color="#b91c1c")
    draw_box(ax, 10.0, 2.3, 1.6, 1.1, "2.6 Approval", "Legitimate\nTransaction", color="#dcfce7", border="#22c55e", text_color="#15803d")

    # Inputs and Stores
    draw_box(ax, 0.6, 1.2, 1.8, 1.0, "Transaction Input", "Single / Batch CSV", color="#f1f5f9", border="#64748b", text_color="#334155")
    draw_box(ax, 5.0, 1.2, 2.0, 1.0, "Trained Model Store", "random_forest.joblib", color="#f1f5f9", border="#64748b", text_color="#334155")

    # Arrows
    draw_arrow(ax, 1.5, 2.2, 1.5, 3.5, label="Raw Input")
    draw_arrow(ax, 2.4, 4.15, 2.8, 4.15, label="Valid")
    draw_arrow(ax, 4.6, 4.15, 5.0, 4.15, label="Scaled")
    draw_arrow(ax, 6.0, 2.2, 6.0, 3.5, label="Trained Weights")
    draw_arrow(ax, 7.0, 4.15, 7.5, 4.15, label="Probability")
    draw_arrow(ax, 9.5, 4.4, 10.0, 4.8, label="P >= Cutoff")
    draw_arrow(ax, 9.5, 3.8, 10.0, 3.0, label="P < Cutoff")

    plt.tight_layout()
    plt.savefig("diagrams/dfd_level_2.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved diagrams/dfd_level_2.png")


def generate_system_flowchart():
    fig, ax = plt.subplots(figsize=(11, 10), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(5.5, 11.5, "End-to-End System Flowchart", ha="center", va="center", fontsize=15, fontweight="bold", color="#1a252c")
    ax.text(5.5, 11.1, "Operational Flow from Transaction Ingestion to Fraud Classification", ha="center", va="center", fontsize=9, color="#6c757d")

    # Flowchart Nodes (Oval for Start/End, Rect for Process, Diamond for Decision)
    def draw_oval(ax, x, y, w, h, text, color="#dbeafe", border="#2563eb"):
        patch = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.35", ec=border, fc=color, lw=1.5, zorder=3)
        ax.add_patch(patch)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1e3a8a", zorder=4)

    def draw_proc(ax, x, y, w, h, title, subtitle="", color="#f0fdf4", border="#16a34a"):
        patch = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.1", ec=border, fc=color, lw=1.5, zorder=3)
        ax.add_patch(patch)
        if subtitle:
            ax.text(x+w/2, y+h/2+0.12, title, ha="center", va="center", fontsize=9, fontweight="bold", color="#14532d", zorder=4)
            ax.text(x+w/2, y+h/2-0.15, subtitle, ha="center", va="center", fontsize=7.5, color="#4b5563", zorder=4)
        else:
            ax.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=9, fontweight="bold", color="#14532d", zorder=4)

    def draw_decision(ax, x, y, size, text, color="#fef9c3", border="#ca8a04"):
        diamond = patches.RegularPolygon((x, y), numVertices=4, radius=size, orientation=0, fc=color, ec=border, lw=1.5, zorder=3)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha="center", va="center", fontsize=8, fontweight="bold", color="#713f12", zorder=4)

    # Sequence of Steps
    draw_oval(ax, 4.3, 10.2, 2.4, 0.6, "Start Application")
    draw_proc(ax, 3.8, 9.0, 3.4, 0.8, "Load Dataset / Transaction Stream", "Read CSV / Streamlit Upload")
    draw_decision(ax, 5.5, 7.8, 0.9, "Schema &\nTypes Valid?")
    draw_proc(ax, 3.8, 5.8, 3.4, 0.8, "Preprocess Features", "RobustScaler on Time & Amount")
    draw_proc(ax, 3.8, 4.4, 3.4, 0.8, "Execute Random Forest Inference", "Compute P(Fraud) = tree vote fraction")
    draw_decision(ax, 5.5, 3.0, 0.9, "P(Fraud) >=\nThreshold?")
    draw_proc(ax, 1.2, 1.7, 3.2, 0.8, "Legitimate Verdict", "Status: Approved", color="#dcfce7", border="#22c55e")
    draw_proc(ax, 6.6, 1.7, 3.2, 0.8, "Fraud Alert Verdict", "Status: Flagged for Review", color="#fee2e2", border="#ef4444")
    draw_oval(ax, 4.3, 0.4, 2.4, 0.6, "End / Log Result")

    # Error handling branch
    draw_proc(ax, 8.2, 7.4, 2.3, 0.8, "Log Error & Alert User", "Missing/Invalid Schema", color="#fef2f2", border="#b91c1c")

    # Flow arrows
    draw_arrow(ax, 5.5, 10.2, 5.5, 9.8)
    draw_arrow(ax, 5.5, 9.0, 5.5, 8.7)
    draw_arrow(ax, 5.5, 6.9, 5.5, 6.6, label="Yes")
    draw_arrow(ax, 6.4, 7.8, 8.2, 7.8, label="No")
    draw_arrow(ax, 9.35, 7.4, 9.35, 0.7, rad=-0.2)
    draw_arrow(ax, 9.35, 0.7, 6.7, 0.7)
    draw_arrow(ax, 5.5, 5.8, 5.5, 5.2)
    draw_arrow(ax, 5.5, 4.4, 5.5, 3.9)
    draw_arrow(ax, 4.6, 3.0, 2.8, 2.5, label="No (Low Risk)")
    draw_arrow(ax, 6.4, 3.0, 8.2, 2.5, label="Yes (High Risk)")
    draw_arrow(ax, 2.8, 1.7, 4.6, 0.7)
    draw_arrow(ax, 8.2, 1.7, 6.4, 0.7)

    plt.tight_layout()
    plt.savefig("diagrams/system_flowchart.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved diagrams/system_flowchart.png")


if __name__ == "__main__":
    generate_system_architecture()
    generate_dfd_level_0()
    generate_dfd_level_1()
    generate_dfd_level_2()
    generate_system_flowchart()
    print("All 5 diagrams successfully generated in diagrams/!")
