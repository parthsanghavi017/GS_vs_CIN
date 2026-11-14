#!/usr/bin/env python3
"""
Phase 3: Machine Learning Classification & Biomarker Discovery
FOCUS: Pure RNA Gene Expression & Transcriptomic Features (Zero CNA / CNV)
STAD TCGA: Genomically Stable (GS) vs. Chromosomal Instability (CIN)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    f1_score, roc_curve, precision_recall_curve, confusion_matrix
)
import shap

# Configure aesthetics
sns.set_theme(style="ticks", font="sans-serif")
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.5
})

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MAIN_RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def evaluate_ml_pipelines(X_train_df, X_test_df, y_tr_series, y_te_series, exp_name="Pure mRNA"):
    """Benchmark 5 ML classifiers with SMOTE and 10-fold CV on train set, then evaluate on test set."""
    scale_pos_ratio = (y_tr_series == 0).sum() / (y_tr_series == 1).sum()

    models = {
        "Random Forest": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight="balanced"))
        ]),
        "XGBoost": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42,
                                 scale_pos_weight=scale_pos_ratio, eval_metric="logloss"))
        ]),
        "LightGBM": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42,
                                  scale_pos_weight=scale_pos_ratio, verbose=-1))
        ]),
        "Logistic Regression": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42))
        ]),
        "Support Vector Machine": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
        ])
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []
    fitted_dict = {}

    for name, pipeline in models.items():
        cv_roc, cv_pr, cv_bal_acc = [], [], []
        for train_idx, val_idx in skf.split(X_train_df, y_tr_series):
            X_tr, y_tr = X_train_df.iloc[train_idx], y_tr_series.iloc[train_idx]
            X_val, y_val = X_train_df.iloc[val_idx], y_tr_series.iloc[val_idx]
            
            pipeline.fit(X_tr, y_tr)
            y_prob = pipeline.predict_proba(X_val)[:, 1]
            y_pred = pipeline.predict(X_val)
            
            cv_roc.append(roc_auc_score(y_val, y_prob))
            cv_pr.append(average_precision_score(y_val, y_prob))
            cv_bal_acc.append(balanced_accuracy_score(y_val, y_pred))
            
        # Fit on full training set
        pipeline.fit(X_train_df, y_tr_series)
        fitted_dict[name] = pipeline
        
        # Evaluate on holdout test set
        y_test_prob = pipeline.predict_proba(X_test_df)[:, 1]
        y_test_pred = pipeline.predict(X_test_df)
        
        test_roc = roc_auc_score(y_te_series, y_test_prob)
        test_pr = average_precision_score(y_te_series, y_test_prob)
        test_bal_acc = balanced_accuracy_score(y_te_series, y_test_pred)
        test_f1 = f1_score(y_te_series, y_test_pred)
        
        metrics_list.append({
            "Experiment": exp_name,
            "Model": name,
            "10CV_ROC_AUC": np.mean(cv_roc),
            "10CV_ROC_AUC_std": np.std(cv_roc),
            "10CV_PR_AUC": np.mean(cv_pr),
            "10CV_Bal_Acc": np.mean(cv_bal_acc),
            "Test_ROC_AUC": test_roc,
            "Test_PR_AUC": test_pr,
            "Test_Bal_Acc": test_bal_acc,
            "Test_F1": test_f1
        })

    return pd.DataFrame(metrics_list), fitted_dict


def plot_roc_pr(fitted_dict, X_test, y_test, title_prefix, save_prefix):
    """Plot ROC and Precision-Recall curves."""
    colors = {
        "Random Forest": "#2b5c8f",
        "XGBoost": "#d95f02",
        "LightGBM": "#7570b3",
        "Logistic Regression": "#1b9e77",
        "Support Vector Machine": "#e7298a"
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # ROC Curves
    for name, pipeline in fitted_dict.items():
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors[name], linewidth=2)

    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.7)
    ax1.set_title(f"ROC Curves ({title_prefix})", pad=15, fontweight="bold", fontsize=13)
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # PR Curves
    for name, pipeline in fitted_dict.items():
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax2.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=colors[name], linewidth=2)

    ax2.axhline(y_test.mean(), linestyle="--", color="#666666", linewidth=1.2, label=f"Baseline ({y_test.mean():.2f})")
    ax2.set_title(f"Precision-Recall Curves ({title_prefix})", pad=15, fontweight="bold", fontsize=13)
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_roc_pr.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_roc_pr.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(fitted_dict, X_test, y_test, title_prefix, save_prefix):
    """Plot confusion matrices for all 5 models on the holdout test set."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=300)
    
    for ax, (name, pipeline) in zip(axes, fitted_dict.items()):
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["GS (0)", "CIN (1)"], yticklabels=["GS (0)", "CIN (1)"])
        ax.set_title(name, fontweight="bold", fontsize=11)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle(f"Holdout Confusion Matrices ({title_prefix})", y=1.05, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_confusion_matrices.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_confusion_matrices.pdf"), bbox_inches="tight")
    plt.close(fig)


def analyze_shap_and_genes(pipeline, X_train, X_test, feature_names, y_test, dge_df, save_prefix):
    """Compute SHAP feature importances and merge with PyDESeq2 statistics to pinpoint differentiating genes."""
    rf_clf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]

    X_test_scaled = scaler.transform(X_test)
    explainer = shap.TreeExplainer(rf_clf)
    shap_vals = explainer(pd.DataFrame(X_test_scaled, columns=feature_names))

    vals = shap_vals.values[:, :, 1] if len(shap_vals.shape) == 3 else shap_vals.values
    mean_abs_shap = np.abs(vals).mean(axis=0)

    # Determine correlation / direction with subtype
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    # Clean gene name
    df_importance["Gene_Symbol"] = df_importance["Feature"].apply(
        lambda x: x.replace("mRNA_", "").replace("TF_", "")
    )
    df_importance["Feature_Type"] = df_importance["Feature"].apply(
        lambda x: "Transcription Factor" if x.startswith("TF_") else "mRNA Gene Expression"
    )

    # Merge with DGE statistics if available
    if dge_df is not None and "gene_name" in dge_df.columns:
        df_merged = df_importance.merge(
            dge_df[["gene_name", "log2FoldChange", "padj", "stat"]],
            left_on="Gene_Symbol", right_on="gene_name", how="left"
        ).drop(columns=["gene_name"])
        df_merged["Enriched_Subtype"] = df_merged["log2FoldChange"].apply(
            lambda lfc: "CIN (Upregulated)" if lfc > 0 else "GS (Upregulated)"
        )
    else:
        df_merged = df_importance

    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, f"{save_prefix}_shap_gene_importances.csv")
    df_merged.to_csv(csv_path, index=False)
    print(f"💾 Saved SHAP gene importances to {csv_path}")

    # Plot top 20 genes SHAP Barplot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    top20 = df_merged.head(20).sort_values("Mean_Abs_SHAP", ascending=True)
    
    colors = ["#2b5c8f" if t == "mRNA Gene Expression" else "#7570b3" for t in top20["Feature_Type"]]
    bars = ax.barh(top20["Feature"], top20["Mean_Abs_SHAP"], color=colors, height=0.65)
    
    ax.set_title(f"Top 20 Differentiating Genes by SHAP ({save_prefix})", pad=15, fontweight="bold", fontsize=13)
    ax.set_xlabel("Mean |SHAP Value| (Predictive Impact)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2b5c8f", label="mRNA Gene Expression"),
        Patch(facecolor="#7570b3", label="TF Activity (DoRothEA)")
    ]
    ax.legend(handles=[el for el in legend_elements if any(t in top20["Feature_Type"].values for t in [el.get_label()])],
              loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_top_shap_genes.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_top_shap_genes.pdf"), bbox_inches="tight")
    plt.close(fig)

    return df_merged


def plot_top_gene_boxplots(df_raw_features, top_genes, save_prefix):
    """Plot expression distributions of top discriminating genes across GS vs CIN."""
    num_genes = min(len(top_genes), 6)
    genes_to_plot = top_genes[:num_genes]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    axes = axes.flatten()

    for i, gene_col in enumerate(genes_to_plot):
        ax = axes[i]
        clean_name = gene_col.replace("mRNA_", "")
        
        sns.boxplot(
            data=df_raw_features,
            x="MOLECULAR_SUBTYPE",
            y=gene_col,
            palette={"GS": "#2b5c8f", "CIN": "#d95f02"},
            width=0.4,
            ax=ax
        )
        sns.stripplot(
            data=df_raw_features,
            x="MOLECULAR_SUBTYPE",
            y=gene_col,
            color="#222222",
            alpha=0.4,
            jitter=0.2,
            size=4,
            ax=ax
        )
        ax.set_title(f"{clean_name} Expression", fontweight="bold", fontsize=12)
        ax.set_xlabel("Subtype", fontsize=10)
        ax.set_ylabel("Standardized z-score", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Top Differentiating Genes: Expression Distributions (GS vs. CIN)", y=1.02, fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_gene_distributions.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_gene_distributions.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("🚀 STARTING PHASE 3: PURE RNA & GENE EXPRESSION CLASSIFIER")
    print("   Removing ALL CNA and CNV features")
    print("=" * 70)

    # Load PyDESeq2 differential expression results for annotation
    dge_file = os.path.join(MAIN_RESULTS_DIR, "dge_deseq2_gs_vs_cin.csv")
    dge_df = pd.read_csv(dge_file) if os.path.exists(dge_file) else None
    if dge_df is not None:
        print(f"📖 Loaded PyDESeq2 DGE reference: {len(dge_df)} genes")

    # =========================================================================
    # EXPERIMENT A: PURE mRNA GENE EXPRESSION ONLY (46 GENES, ZERO CNA/CNV/TF)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔬 EXPERIMENT A: PURE mRNA GENE EXPRESSION ONLY (NO CNA, NO CNV, NO TF)")
    print("=" * 70)

    mrna_file = os.path.join(DATA_DIR, "mrna_only_feature_matrix.csv")
    df_mrna = pd.read_csv(mrna_file, index_col=0)
    
    X_mrna = df_mrna.drop(columns=["MOLECULAR_SUBTYPE"])
    y = (df_mrna["MOLECULAR_SUBTYPE"] == "CIN").astype(int)

    print(f"📊 Dataset: {X_mrna.shape[0]} samples x {X_mrna.shape[1]} pure mRNA genes")
    print(f"   - Target Distribution: CIN={sum(y==1)} ({y.mean()*100:.1f}%), GS={sum(y==0)} ({(1-y.mean())*100:.1f}%)")

    # Stratified Train-Test Split (80% Train, 20% Holdout Test)
    X_tr_mrna, X_te_mrna, y_tr, y_te = train_test_split(
        X_mrna, y, test_size=0.20, stratify=y, random_state=42
    )

    metrics_mrna, fitted_mrna = evaluate_ml_pipelines(
        X_tr_mrna, X_te_mrna, y_tr, y_te, exp_name="Pure mRNA (46 Genes)"
    )

    # Save metrics
    metrics_mrna_path = os.path.join(RESULTS_DIR, "ml_pure_mrna_only_metrics.csv")
    metrics_mrna.to_csv(metrics_mrna_path, index=False)
    print("\n🏆 Experiment A Results (Pure mRNA Genes Only):")
    print(metrics_mrna[["Model", "10CV_ROC_AUC", "10CV_PR_AUC", "10CV_Bal_Acc", "Test_ROC_AUC", "Test_PR_AUC", "Test_Bal_Acc", "Test_F1"]].to_string())

    # Visualizations
    plot_roc_pr(fitted_mrna, X_te_mrna, y_te, "Pure mRNA Genes (No CNA/CNV)", "pure_mrna")
    plot_confusion_matrices(fitted_mrna, X_te_mrna, y_te, "Pure mRNA Genes (No CNA/CNV)", "pure_mrna")
    
    # SHAP & Differentiating Gene Identification
    shap_mrna = analyze_shap_and_genes(
        fitted_mrna["Random Forest"], X_tr_mrna, X_te_mrna, X_mrna.columns, y_te, dge_df, "pure_mrna"
    )

    # Plot top differentiating gene boxplots
    top_mrna_genes = shap_mrna["Feature"].tolist()
    plot_top_gene_boxplots(df_mrna, top_mrna_genes, "pure_mrna")

    # =========================================================================
    # EXPERIMENT B: RNA-BASED MULTI-OMICS (mRNA + DoRothEA TF ACTIVITY, NO CNA/CNV)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔬 EXPERIMENT B: mRNA + TF REGULATORY ACTIVITY (ZERO CNA / CNV)")
    print("=" * 70)

    mrna_tf_file = os.path.join(DATA_DIR, "mrna_tf_feature_matrix.csv")
    df_mrna_tf = pd.read_csv(mrna_tf_file, index_col=0)
    
    X_mrna_tf = df_mrna_tf.drop(columns=["MOLECULAR_SUBTYPE"])
    
    print(f"📊 Dataset: {X_mrna_tf.shape[0]} samples x {X_mrna_tf.shape[1]} RNA features (46 mRNA + 30 TFs)")

    X_tr_tf, X_te_tf, _, _ = train_test_split(
        X_mrna_tf, y, test_size=0.20, stratify=y, random_state=42
    )

    metrics_tf, fitted_tf = evaluate_ml_pipelines(
        X_tr_tf, X_te_tf, y_tr, y_te, exp_name="mRNA + TF (No CNA/CNV)"
    )

    # Save metrics
    metrics_tf_path = os.path.join(RESULTS_DIR, "ml_mrna_tf_metrics.csv")
    metrics_tf.to_csv(metrics_tf_path, index=False)
    print("\n🏆 Experiment B Results (mRNA + TF Activity):")
    print(metrics_tf[["Model", "10CV_ROC_AUC", "10CV_PR_AUC", "10CV_Bal_Acc", "Test_ROC_AUC", "Test_PR_AUC", "Test_Bal_Acc", "Test_F1"]].to_string())

    plot_roc_pr(fitted_tf, X_te_tf, y_te, "mRNA + TF Activity (No CNA/CNV)", "mrna_tf")
    plot_confusion_matrices(fitted_tf, X_te_tf, y_te, "mRNA + TF Activity (No CNA/CNV)", "mrna_tf")
    
    shap_tf = analyze_shap_and_genes(
        fitted_tf["Random Forest"], X_tr_tf, X_te_tf, X_mrna_tf.columns, y_te, dge_df, "mrna_tf"
    )

    # =========================================================================
    # COMPARATIVE BENCHMARK: PURE mRNA vs. mRNA+TF vs. FULL MULTI-OMICS
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 COMPILING COMPARATIVE ABLATION ANALYSIS")
    print("=" * 70)

    # Load original full model metrics if available
    orig_file = os.path.join(MAIN_RESULTS_DIR, "ml_model_comparison_metrics.csv")
    df_combined_list = [metrics_mrna, metrics_tf]
    if os.path.exists(orig_file):
        df_orig = pd.read_csv(orig_file)
        df_orig["Experiment"] = "Full Multi-Omics (With CNA/CNV)"
        df_combined_list.append(df_orig)

    df_all_comparisons = pd.concat(df_combined_list, axis=0).reset_index(drop=True)
    all_comp_path = os.path.join(RESULTS_DIR, "all_experiments_comparison_metrics.csv")
    df_all_comparisons.to_csv(all_comp_path, index=False)
    print(f"💾 Saved complete comparison to {all_comp_path}")

    # Comparative Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=300)
    
    palette = {
        "Pure mRNA (46 Genes)": "#2b5c8f",
        "mRNA + TF (No CNA/CNV)": "#7570b3",
        "Full Multi-Omics (With CNA/CNV)": "#d95f02"
    }

    sns.barplot(
        data=df_all_comparisons,
        x="Model",
        y="10CV_ROC_AUC",
        hue="Experiment",
        palette=palette,
        ax=ax1
    )
    ax1.set_title("10-Fold CV ROC-AUC Across Feature Regimes", pad=15, fontweight="bold", fontsize=12)
    ax1.set_ylabel("10-Fold CV ROC-AUC")
    ax1.set_ylim(0.80, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")

    sns.barplot(
        data=df_all_comparisons,
        x="Model",
        y="Test_ROC_AUC",
        hue="Experiment",
        palette=palette,
        ax=ax2
    )
    ax2.set_title("Holdout Test ROC-AUC Across Feature Regimes", pad=15, fontweight="bold", fontsize=12)
    ax2.set_ylabel("Holdout Test ROC-AUC")
    ax2.set_ylim(0.80, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "cross_regime_model_comparison.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, "cross_regime_model_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)

    print("\n✅ PHASE 3 PURE RNA ANALYSIS COMPLETE!")
    print(f"📁 All metrics, rankings, and figures saved in {BASE_DIR}")


if __name__ == "__main__":
    main()
