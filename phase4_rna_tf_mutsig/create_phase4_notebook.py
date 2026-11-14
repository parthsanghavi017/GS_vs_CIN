#!/usr/bin/env python3
"""
Script to create the Phase 4 Jupyter notebook programmatically.
Run with:  conda run -n gs_vs_cin python create_phase4_notebook.py
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# ─── kernel metadata ────────────────────────────────────────────────────────
nb.metadata = {
    "kernelspec": {
        "display_name": "gs_vs_cin",
        "language": "python",
        "name": "gs_vs_cin"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

cells = []

# ============================================================
# CELL 0 — Title markdown
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""\
# Phase 4: mRNA + Transcription Factor Activity + COSMIC Mutational Signatures
## TCGA-STAD: Genomically Stable (GS) vs. Chromosomal Instability (CIN)

**Experiment:** Can we classify GS vs. CIN using only:
- 46 DESeq2 significant mRNA transcripts  
- 30 DoRothEA transcription factor regulon activities  
- 5 COSMIC SBS mutational signatures (SBS1, SBS3, SBS5, SBS17, SBS2_13_APOBEC)  
- = **81 total features, ZERO CNA / ZERO CNV**

**Key questions addressed:**
1. Does adding mutational signatures (SBS) on top of mRNA + TF activity improve classification?
2. Which features drive GS vs. CIN separation — and across which biological modalities?
3. Can COSMIC SBS signature enrichment patterns differentiate the subtypes?

---
"""))

# ============================================================
# CELL 1 — Imports
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## 1. Setup & Imports"))
cells.append(nbf.v4.new_code_cell("""\
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier, DMatrix
from lightgbm import LGBMClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    f1_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)

# ── Aesthetics ────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", font="sans-serif")
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.5
})

PALETTE      = {"GS": "#2b7bba", "CIN": "#d62728"}
MODEL_COLORS = {
    "Random Forest":          "#1f77b4",
    "XGBoost":                "#ff7f0e",
    "LightGBM":               "#2ca02c",
    "Logistic Regression":    "#9467bd",
    "Support Vector Machine": "#d62728"
}
print("✅ Imports complete")
"""))

# ============================================================
# CELL 2 — Paths
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## 2. Paths & Feature Definitions"))
cells.append(nbf.v4.new_code_cell("""\
# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath("__file__"))
DATA_DIR     = os.path.join(BASE_DIR, "data")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
FIGURES_DIR  = os.path.join(BASE_DIR, "figures")
MAIN_RESULTS = os.path.join(BASE_DIR, "..", "results")
PHASE3_DATA  = os.path.join(BASE_DIR, "..", "phase3_rna_only", "data")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Feature Definitions ───────────────────────────────────────────────────────
# 46 DESeq2 significant mRNA transcripts (from Phase 3)
MRNA_GENES = [
    "SEPTIN6", "AURKA", "RAE1", "RASSF2", "YTHDF1", "SH3BP5",
    "TSPAN11", "GRB7", "CSE1L", "ANLN", "FLI1", "DDX27", "ZEB2",
    "UBE2C", "MOB3A", "MFAP2", "DLGAP5", "FBXO5", "HJURP",
    "CENPF", "KIF2C", "BIRC5", "NDC80", "KIF20A", "CEP55",
    "TPX2", "MKI67", "CCNB1", "CDC20", "MELK", "EXO1", "KIF4A",
    "PTTG1", "NCAPG", "PLK1", "BUB1", "CDCA8", "ESPL1", "MAD2L1",
    "CDKN3", "OIP5", "RACGAP1", "PRC1", "TTK", "SHCBP1", "SPC25"
]

# 30 DoRothEA TF regulons (from Phase 3 feature matrix)
TF_REGULONS = [
    "E2F1", "E2F2", "E2F3", "E2F4", "E2F7",
    "FOXM1", "MYC", "MYCN", "MYB", "MYBL2",
    "TP53", "TP63", "TP73",
    "GLI1", "GLI2",
    "ZEB1", "ZEB2",
    "STAT3", "STAT5A", "STAT5B",
    "NFKB1", "NFKB2", "RELA",
    "CEBPA", "CEBPB",
    "HIF1A", "EPAS1",
    "YBX1", "MXI1", "MNT"
]

# 5 COSMIC SBS mutational signatures
SBS_SIGS = [
    "MutSig_SBS1_Aging",
    "MutSig_SBS3_HRD",
    "MutSig_SBS5_Clock",
    "MutSig_SBS17_Gastric_5FU",
    "MutSig_SBS2_13_APOBEC"
]

SBS_DISPLAY_NAMES = {
    "MutSig_SBS1_Aging":       "SBS1 — Aging Clock",
    "MutSig_SBS3_HRD":         "SBS3 — DNA Repair (HRD)",
    "MutSig_SBS5_Clock":       "SBS5 — Clock-like",
    "MutSig_SBS17_Gastric_5FU":"SBS17 — Gastric/5-FU",
    "MutSig_SBS2_13_APOBEC":   "SBS2_13 — APOBEC ★",
}

print(f"mRNA genes:       {len(MRNA_GENES)}")
print(f"TF regulons:      {len(TF_REGULONS)}")
print(f"SBS signatures:   {len(SBS_SIGS)}")
print(f"Total features:   {len(MRNA_GENES) + len(TF_REGULONS) + len(SBS_SIGS)}")
"""))

# ============================================================
# CELL 3 — Load Data
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""\
## 3. Load & Integrate Data

We merge:
1. **Phase 2 integrated feature store** (holds SBS mutational signature exposures)
2. **Phase 3 mRNA + TF feature matrix** (mRNA gene expression + DoRothEA TF activities)
"""))
cells.append(nbf.v4.new_code_cell("""\
# ── Load Phase 2 feature store ────────────────────────────────────────────────
feature_store = os.path.join(MAIN_RESULTS, "phase2_integrated_feature_store.csv")
print(f"📖 Loading feature store: {feature_store}")
df_full = pd.read_csv(feature_store, index_col=0)
print(f"   Loaded: {df_full.shape[0]} patients × {df_full.shape[1]} features")

# ── Load Phase 3 TF feature matrix ───────────────────────────────────────────
mrna_tf_file = os.path.join(PHASE3_DATA, "mrna_tf_feature_matrix.csv")
df_mrna_tf   = pd.read_csv(mrna_tf_file, index_col=0)
mrna_tf_cols = [c for c in df_mrna_tf.columns if c != "MOLECULAR_SUBTYPE"]
mrna_cols      = [c for c in mrna_tf_cols if c.startswith("mRNA_")]
tf_cols_phase3 = [c for c in mrna_tf_cols if not c.startswith("mRNA_")]

print(f"\\n   mRNA columns from Phase 3 matrix: {len(mrna_cols)}")
print(f"   TF  columns from Phase 3 matrix : {len(tf_cols_phase3)}")

# ── Identify SBS columns ──────────────────────────────────────────────────────
sbs_cols_available = [c for c in df_full.columns if c.startswith("MutSig_")]
print(f"   SBS columns in feature store   : {len(sbs_cols_available)} → {sbs_cols_available}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Merge datasets ─────────────────────────────────────────────────────────────
target_col = "SUBTYPE_CIN_VS_GS"
label_col  = "MOLECULAR_SUBTYPE"

# Normalize phase3 index (strip '-01' suffix) to match feature store format
df_mrna_tf_norm = df_mrna_tf.copy()
df_mrna_tf_norm.index = df_mrna_tf_norm.index.str.replace(r"-\\d{2}$", "", regex=True)

# Find common patients
common_patients = df_mrna_tf_norm.index.intersection(df_full.index)
print(f"   Common patients (after index normalization): {len(common_patients)}")

df_mrna_tf_aligned = df_mrna_tf_norm.loc[common_patients].copy()
df_sbs_aligned     = df_full.loc[common_patients, sbs_cols_available + [label_col]].copy()

# Merge
df_combined = df_mrna_tf_aligned.join(df_sbs_aligned[sbs_cols_available], how="inner")
print(f"\\n✅ Combined dataset: {df_combined.shape[0]} patients × {df_combined.shape[1]} columns")

# Add subtype string for plotting
df_combined["SUBTYPE"] = df_sbs_aligned[label_col]

# Features + target
X_all = df_combined.drop(columns=["MOLECULAR_SUBTYPE", "SUBTYPE"])
y_all = (df_combined["MOLECULAR_SUBTYPE"] == "CIN").astype(int)

print(f"   Feature matrix: {X_all.shape[0]} × {X_all.shape[1]}")
print(f"   CIN: {(y_all==1).sum()}, GS: {(y_all==0).sum()}")

# Save combined feature matrix
X_all.join(df_combined[["MOLECULAR_SUBTYPE"]]).to_csv(
    os.path.join(DATA_DIR, "mrna_tf_sbs_feature_matrix.csv"))
print(f"   💾 Feature matrix saved → data/mrna_tf_sbs_feature_matrix.csv")
"""))

# ============================================================
# CELL 4 — Train/Test Split
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## 4. Train / Test Split"))
cells.append(nbf.v4.new_code_cell("""\
# Same seed as Phase 3 for fair comparison
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=42
)
print(f"📊 Train: {len(X_tr)} | Test: {len(X_te)}")
print(f"   Train — CIN: {(y_tr==1).sum()}, GS: {(y_tr==0).sum()}")
print(f"   Test  — CIN: {(y_te==1).sum()}, GS: {(y_te==0).sum()}")
"""))

# ============================================================
# CELL 5 — SBS Differential Analysis
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""\
## 5. COSMIC SBS Mutational Signature — Differential Analysis

Before ML, we examine whether SBS signature exposures significantly differ between GS and CIN subtypes.
"""))
cells.append(nbf.v4.new_code_cell("""\
def plot_sbs_distributions(df_combined, save_prefix):
    sbs_cols = [c for c in df_combined.columns if c.startswith("MutSig_")]
    n = len(sbs_cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), dpi=150)
    if n == 1: axes = [axes]

    stat_rows = []
    for ax, col in zip(axes, sbs_cols):
        gs_vals  = df_combined.loc[df_combined["SUBTYPE"] == "GS",  col]
        cin_vals = df_combined.loc[df_combined["SUBTYPE"] == "CIN", col]
        t_stat, p_val = stats.mannwhitneyu(cin_vals, gs_vals, alternative="two-sided")

        data_plot = pd.DataFrame({
            "Exposure": pd.concat([gs_vals, cin_vals]),
            "Subtype":  ["GS"] * len(gs_vals) + ["CIN"] * len(cin_vals)
        })
        sns.boxplot(data=data_plot, x="Subtype", y="Exposure", hue="Subtype",
                    palette=PALETTE, ax=ax, order=["GS", "CIN"], width=0.5,
                    linewidth=1.5, legend=False)
        sns.stripplot(data=data_plot, x="Subtype", y="Exposure", hue="Subtype",
                      palette=PALETTE, ax=ax, order=["GS", "CIN"], size=3,
                      alpha=0.5, jitter=True, legend=False)

        display_name = SBS_DISPLAY_NAMES.get(col, col)
        ax.set_title(f"{display_name}\\np = {p_val:.2e}", fontweight="bold", fontsize=10)
        ax.set_xlabel(""); ax.set_ylabel("Mutational Exposure")
        ax.grid(True, axis="y", alpha=0.3)

        stat_rows.append({
            "Signature": col,
            "CIN_Mean": cin_vals.mean(), "GS_Mean": gs_vals.mean(),
            "Delta_CIN_minus_GS": cin_vals.mean() - gs_vals.mean(),
            "MannWhitney_U": t_stat, "p_value": p_val,
            "Significant_p05": p_val < 0.05
        })

    plt.suptitle("COSMIC SBS Mutational Signature Exposures: GS vs. CIN",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_sbs_distributions.png"),
                bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_sbs_distributions.pdf"),
                bbox_inches="tight")
    plt.show()
    print(f"  📊 Saved SBS distributions → {save_prefix}_sbs_distributions.png")
    return pd.DataFrame(stat_rows)

sbs_stat_df = plot_sbs_distributions(df_combined, "phase4")
sbs_stat_df.to_csv(os.path.join(RESULTS_DIR, "sbs_differential_statistics.csv"), index=False)
print("\\n💾 SBS stats saved → results/sbs_differential_statistics.csv")
print(sbs_stat_df[["Signature","CIN_Mean","GS_Mean","Delta_CIN_minus_GS","p_value","Significant_p05"]].to_string())
"""))

# ============================================================
# CELL 6 — ML Benchmark
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""\
## 6. Machine Learning Benchmark

Five classifiers are trained with:
- **SMOTE** oversampling (to handle class imbalance: 125 CIN vs 52 GS)
- **10-fold stratified cross-validation** on training data
- **Holdout test evaluation** on 20% held-out set

Feature regime: mRNA + TF + COSMIC SBS (81 features)
"""))
cells.append(nbf.v4.new_code_cell("""\
def evaluate_ml_pipelines(X_train_df, X_test_df, y_tr, y_te, exp_name="mRNA+TF+SBS"):
    \"\"\"Train 5 classifiers with SMOTE+10-fold CV and evaluate on holdout test set.\"\"\"
    scale_pos_ratio = (y_tr == 0).sum() / (y_tr == 1).sum()

    models = {
        "Random Forest": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=42)),
            ("clf",    RandomForestClassifier(n_estimators=150, max_depth=6,
                                               random_state=42, class_weight="balanced"))
        ]),
        "XGBoost": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=42)),
            ("clf",    XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                      random_state=42, scale_pos_weight=scale_pos_ratio,
                                      eval_metric="logloss"))
        ]),
        "LightGBM": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=42)),
            ("clf",    LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                       random_state=42, scale_pos_weight=scale_pos_ratio,
                                       verbose=-1))
        ]),
        "Logistic Regression": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=42)),
            ("clf",    LogisticRegression(C=0.5, class_weight="balanced",
                                          max_iter=1000, random_state=42))
        ]),
        "Support Vector Machine": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=42)),
            ("clf",    SVC(kernel="rbf", probability=True,
                           class_weight="balanced", random_state=42))
        ])
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list, fitted_dict = [], {}

    for name, pipe in models.items():
        cv_roc, cv_pr, cv_bal = [], [], []
        for tr_idx, val_idx in skf.split(X_train_df, y_tr):
            Xtr, ytr = X_train_df.iloc[tr_idx], y_tr.iloc[tr_idx]
            Xvl, yvl = X_train_df.iloc[val_idx], y_tr.iloc[val_idx]
            pipe.fit(Xtr, ytr)
            yp  = pipe.predict_proba(Xvl)[:, 1]
            ypd = pipe.predict(Xvl)
            cv_roc.append(roc_auc_score(yvl, yp))
            cv_pr.append(average_precision_score(yvl, yp))
            cv_bal.append(balanced_accuracy_score(yvl, ypd))

        # Final fit on full train, evaluate on test
        pipe.fit(X_train_df, y_tr)
        fitted_dict[name] = pipe
        yp_te  = pipe.predict_proba(X_test_df)[:, 1]
        ypd_te = pipe.predict(X_test_df)

        metrics_list.append({
            "Experiment":       exp_name,
            "Model":            name,
            "10CV_ROC_AUC":     np.mean(cv_roc),
            "10CV_ROC_AUC_std": np.std(cv_roc),
            "10CV_PR_AUC":      np.mean(cv_pr),
            "10CV_Bal_Acc":     np.mean(cv_bal),
            "Test_ROC_AUC":     roc_auc_score(y_te, yp_te),
            "Test_PR_AUC":      average_precision_score(y_te, yp_te),
            "Test_Bal_Acc":     balanced_accuracy_score(y_te, ypd_te),
            "Test_F1":          f1_score(y_te, ypd_te),
            "Feature_Count":    X_train_df.shape[1],
            "Modalities":       "mRNA + DoRothEA TFs + COSMIC SBS Mutational Signatures"
        })
        print(f"  ✅ {name:25s} | 10CV AUC={np.mean(cv_roc):.3f} | Test AUC={roc_auc_score(y_te, yp_te):.3f}")

    return pd.DataFrame(metrics_list), fitted_dict

print("=" * 70)
print("🤖 RUNNING ML CLASSIFIERS: mRNA + TF + SBS (81 features)")
print("=" * 70)
metrics_sbs, fitted_sbs = evaluate_ml_pipelines(X_tr, X_te, y_tr, y_te, "mRNA + TF + SBS")

# Save metrics
metrics_sbs.to_csv(os.path.join(RESULTS_DIR, "ml_mrna_tf_sbs_metrics.csv"), index=False)
print("\\n🏆 mRNA + TF + SBS Results:")
print(metrics_sbs[["Model","10CV_ROC_AUC","10CV_ROC_AUC_std","Test_ROC_AUC",
                     "Test_PR_AUC","Test_Bal_Acc","Test_F1"]].to_string())
"""))

# ============================================================
# CELL 7 — ROC / PR Curves
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## 7. ROC & Precision-Recall Curves"))
cells.append(nbf.v4.new_code_cell("""\
def plot_roc_pr(fitted_dict, X_test, y_test, title_prefix, save_prefix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    for name, pipe in fitted_dict.items():
        yp = pipe.predict_proba(X_test)[:, 1]
        color = MODEL_COLORS.get(name, "#333333")

        fpr, tpr, _ = roc_curve(y_test, yp)
        auc = roc_auc_score(y_test, yp)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, yp)
        pr_auc = average_precision_score(y_test, yp)
        ax2.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={pr_auc:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_title(f"ROC Curves\\n{title_prefix}", fontweight="bold", fontsize=12)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.legend(fontsize=8, loc="lower right"); ax1.grid(True, alpha=0.3)

    base_prec = y_test.sum() / len(y_test)
    ax2.axhline(base_prec, color="gray", linestyle="--", lw=1,
                label=f"Baseline ({base_prec:.2f})")
    ax2.set_title(f"Precision-Recall Curves\\n{title_prefix}", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.legend(fontsize=8, loc="lower left"); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_roc_pr.png"),
                bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_roc_pr.pdf"),
                bbox_inches="tight")
    plt.show()
    print(f"  📊 Saved ROC/PR curves → {save_prefix}_roc_pr.png")

plot_roc_pr(fitted_sbs, X_te, y_te, "mRNA + TF + COSMIC SBS (Zero CNA/CNV)", "phase4")
"""))

# ============================================================
# CELL 8 — Confusion Matrices
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## 8. Confusion Matrices"))
cells.append(nbf.v4.new_code_cell("""\
def plot_confusion_matrices(fitted_dict, X_test, y_test, title_prefix, save_prefix):
    n = len(fitted_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=150)

    detail_rows = []
    for ax, (name, pipe) in zip(axes, fitted_dict.items()):
        ypd = pipe.predict(X_test)
        cm  = confusion_matrix(y_test, ypd)
        tn, fp, fn, tp = cm.ravel()

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred GS", "Pred CIN"],
                    yticklabels=["True GS", "True CIN"])
        ax.set_title(f"{name}\\nAcc={((tp+tn)/len(y_test)):.1%}",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

        detail_rows.append({
            "Model": name,
            "TN_GS_Correct": int(tn), "FP_GS_Called_CIN": int(fp),
            "FN_CIN_Called_GS": int(fn), "TP_CIN_Correct": int(tp),
            "Sensitivity_CIN": f"{tp/(tp+fn):.1%}",
            "Specificity_GS":  f"{tn/(tn+fp):.1%}",
            "Overall_Accuracy": f"{(tp+tn)/len(y_test):.1%}"
        })

    plt.suptitle(f"Confusion Matrices — {title_prefix}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_confusion_matrices.png"),
                bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_confusion_matrices.pdf"),
                bbox_inches="tight")
    plt.show()
    print(f"  📊 Saved confusion matrices → {save_prefix}_confusion_matrices.png")
    return pd.DataFrame(detail_rows)

cm_detail = plot_confusion_matrices(fitted_sbs, X_te, y_te,
                                    "mRNA + TF + COSMIC SBS", "phase4")
cm_detail.to_csv(os.path.join(RESULTS_DIR, "confusion_matrix_detail.csv"), index=False)
print("\\nConfusion matrix breakdown:")
print(cm_detail.to_string(index=False))
"""))

# ============================================================
# CELL 9 — SHAP Feature Importance
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""\
## 9. SHAP Feature Importance Analysis

TreeSHAP values computed using XGBoost's native `pred_contribs=True` 
(bypasses the SHAP library's XGBoost 3.x JSON incompatibility).

Features are colour-coded by **biological modality**:
- 🔵 **Blue** — mRNA Gene Expression  
- 🟢 **Green** — TF Regulon Activity (DoRothEA)  
- 🔴 **Red** — COSMIC SBS Mutational Signature
"""))
cells.append(nbf.v4.new_code_cell("""\
def plot_shap_importances(pipe, X_train, X_test, feature_names, save_prefix, top_n=25):
    \"\"\"
    Compute and plot SHAP feature importances using XGBoost native pred_contribs.
    Compatible with XGBoost 3.x (bypasses SHAP library version incompatibility).
    \"\"\"
    try:
        clf    = pipe.named_steps["clf"]
        scaler = pipe.named_steps["scaler"]
        X_te_scaled = scaler.transform(X_test).astype(np.float32)

        # XGBoost native SHAP via pred_contribs=True
        booster = clf.get_booster()
        dmat    = DMatrix(pd.DataFrame(X_te_scaled, columns=feature_names))
        # Shape: (n_samples, n_features + 1) — last col is bias/intercept
        contrib_matrix = booster.predict(dmat, pred_contribs=True)
        vals = contrib_matrix[:, :-1]       # drop bias column

        mean_abs = np.abs(vals).mean(axis=0)
        shap_df  = pd.DataFrame({"Feature": feature_names, "Mean_SHAP": mean_abs})
        shap_df  = shap_df.sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)

        # Colour by modality
        def _modality_color(feat):
            if feat.startswith("MutSig_"):  return "#e31a1c"
            elif feat.startswith("mRNA_"):  return "#1f78b4"
            else:                           return "#33a02c"  # TF

        top    = shap_df.head(top_n)
        colors = [_modality_color(f) for f in top["Feature"]]

        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        ax.barh(top["Feature"][::-1], top["Mean_SHAP"][::-1],
                color=colors[::-1], edgecolor="white", height=0.7)
        ax.set_xlabel("Mean |SHAP Value|  (impact on CIN prediction)", fontsize=11)
        ax.set_title(f"Top {top_n} Feature Importances (SHAP)\\nmRNA + TF + COSMIC SBS",
                     fontweight="bold", fontsize=13)
        ax.grid(True, axis="x", alpha=0.3)

        # Legend for modality colours
        legend_patches = [
            mpatches.Patch(color="#1f78b4", label="mRNA Gene Expression"),
            mpatches.Patch(color="#33a02c", label="TF Regulon Activity (DoRothEA)"),
            mpatches.Patch(color="#e31a1c", label="COSMIC SBS Mutational Signature"),
        ]
        ax.legend(handles=legend_patches, fontsize=9, loc="lower right")

        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_shap_importances.png"),
                    bbox_inches="tight", dpi=200)
        fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_shap_importances.pdf"),
                    bbox_inches="tight")
        plt.show()
        print(f"  📊 Saved SHAP importance plot → {save_prefix}_shap_importances.png")
        return shap_df
    except Exception as e:
        print(f"  ⚠️  SHAP computation failed: {e}")
        return pd.DataFrame()

print("🔍 Computing SHAP importances (XGBoost native TreeSHAP)...")
shap_df = plot_shap_importances(
    fitted_sbs["XGBoost"], X_tr, X_te, X_all.columns.tolist(), "phase4", top_n=25
)
if not shap_df.empty:
    shap_df.to_csv(os.path.join(RESULTS_DIR, "shap_feature_importances.csv"), index=False)
    print("   💾 SHAP importances saved → results/shap_feature_importances.csv")
"""))

# ============================================================
# CELL 10 — SHAP results table
# ============================================================
cells.append(nbf.v4.new_markdown_cell("### 9.1 Top SHAP Features — Tabular View"))
cells.append(nbf.v4.new_code_cell("""\
print("\\n   Top 15 Features by SHAP (XGBoost):")
print(shap_df.head(15)[["Feature","Mean_SHAP"]].to_string(index=False))

print("\\n\\nModality breakdown of top 25 features:")
top25 = shap_df.head(25).copy()
top25["Modality"] = top25["Feature"].apply(lambda f:
    "COSMIC SBS" if f.startswith("MutSig_") else
    ("mRNA Expression" if f.startswith("mRNA_") else "TF Regulon Activity"))
print(top25.groupby("Modality").size().to_string())

# Display full table
display(shap_df.head(25).style.background_gradient(subset=["Mean_SHAP"], cmap="YlOrRd"))
"""))

# ============================================================
# CELL 10b — SHAP Beeswarm
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""\
### 9.2 SHAP Beeswarm Plot — Direction of Feature Effects

The beeswarm shows the **signed** (raw) SHAP value for every patient (dot) and every feature.

- **X-axis > 0** (right) → feature pushes model toward predicting **CIN**
- **X-axis < 0** (left) → feature pushes model toward predicting **GS**
- **Dot colour** → actual value of the feature for that patient
  - 🔴 Red = high feature value | 🔵 Blue = low feature value

This reveals *direction* that the Mean_SHAP bar chart collapses away.
"""))
cells.append(nbf.v4.new_code_cell("""\
def plot_shap_beeswarm(pipe, X_test, feature_names, save_prefix, top_n=25):
    \"\"\"
    Beeswarm of signed SHAP values.
    Each dot = one test-set patient.
    Positive SHAP = pushes toward CIN | Negative SHAP = pushes toward GS.
    Dot colour = raw feature value (red=high, blue=low).
    \"\"\"
    clf    = pipe.named_steps["clf"]
    scaler = pipe.named_steps["scaler"]
    X_te_scaled = scaler.transform(X_test).astype(np.float32)

    from xgboost import DMatrix
    booster = clf.get_booster()
    dmat    = DMatrix(pd.DataFrame(X_te_scaled, columns=feature_names))
    contrib_matrix = booster.predict(dmat, pred_contribs=True)
    vals     = contrib_matrix[:, :-1].astype(np.float32)   # drop bias column
    base_val = contrib_matrix[:, -1].mean()

    # Build shap.Explanation
    explanation = shap.Explanation(
        values        = vals,
        base_values   = np.full(len(X_test), base_val),
        data          = X_te_scaled,
        feature_names = list(feature_names)
    )

    # Subset to top_n by mean |SHAP|
    mean_abs_order  = np.argsort(np.abs(vals).mean(axis=0))[::-1]
    top_idx         = mean_abs_order[:top_n]
    explanation_top = shap.Explanation(
        values        = vals[:, top_idx],
        base_values   = explanation.base_values,
        data          = X_te_scaled[:, top_idx],
        feature_names = [feature_names[i] for i in top_idx]
    )

    plt.figure(figsize=(11, 9), dpi=150)
    shap.plots.beeswarm(explanation_top, max_display=top_n, show=False, color_bar=True)
    plt.title(
        f"SHAP Beeswarm — Top {top_n} Features (XGBoost)\\n"
        "mRNA + TF + COSMIC SBS  |  Positive = CIN driver, Negative = GS driver",
        fontweight="bold", fontsize=12, pad=14
    )
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_shap_beeswarm.png"),
                bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_shap_beeswarm.pdf"),
                bbox_inches="tight")
    plt.show()
    print(f"  📊 Saved SHAP beeswarm → {save_prefix}_shap_beeswarm.png")

print("\\n🐝 Generating SHAP beeswarm plot (Top 25)...")
plot_shap_beeswarm(
    fitted_sbs["XGBoost"], X_te, X_all.columns.tolist(), "phase4", top_n=25
)
"""))

# ============================================================
# CELL 11 — Regime Comparison
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""\
## 10. Three-Way Feature Regime Comparison

Compare classification performance across:
- **Pure mRNA** (46 genes)  
- **mRNA + TF** (76 features)  
- **mRNA + TF + SBS** (81 features)
"""))
cells.append(nbf.v4.new_code_cell("""\
def plot_regime_comparison(metrics_sbs, metrics_tf_only, metrics_mrna_only, save_prefix):
    combined = pd.concat([
        metrics_mrna_only.assign(Regime="Pure mRNA (46 genes)"),
        metrics_tf_only.assign(Regime="mRNA + TF (76 features)"),
        metrics_sbs.assign(Regime="mRNA + TF + SBS (81 features)"),
    ])

    palette_regime = {
        "Pure mRNA (46 genes)":          "#2b5c8f",
        "mRNA + TF (76 features)":       "#7570b3",
        "mRNA + TF + SBS (81 features)": "#e31a1c",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    sns.barplot(data=combined, x="Model", y="Test_ROC_AUC", hue="Regime",
                palette=palette_regime, ax=ax1)
    ax1.set_title("Holdout Test ROC-AUC\\nacross RNA Feature Regimes",
                  fontweight="bold", fontsize=12)
    ax1.set_ylabel("Test ROC-AUC"); ax1.set_ylim(0.78, 1.0)
    ax1.grid(True, axis="y", alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")

    sns.barplot(data=combined, x="Model", y="Test_F1", hue="Regime",
                palette=palette_regime, ax=ax2)
    ax2.set_title("Holdout Test F1-Score\\nacross RNA Feature Regimes",
                  fontweight="bold", fontsize=12)
    ax2.set_ylabel("Test F1-Score"); ax2.set_ylim(0.78, 1.0)
    ax2.grid(True, axis="y", alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_regime_comparison.png"),
                bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_regime_comparison.pdf"),
                bbox_inches="tight")
    plt.show()
    print(f"  📊 Saved regime comparison → {save_prefix}_regime_comparison.png")

# Load Phase 3 metrics for comparison
BASE_DIR_PHASE3 = os.path.join(BASE_DIR, "..", "phase3_rna_only")
mrna_metrics_path = os.path.join(BASE_DIR_PHASE3, "results", "ml_pure_mrna_only_metrics.csv")
tf_metrics_path   = os.path.join(BASE_DIR_PHASE3, "results", "ml_mrna_tf_metrics.csv")

if os.path.exists(mrna_metrics_path) and os.path.exists(tf_metrics_path):
    metrics_mrna_only = pd.read_csv(mrna_metrics_path)
    metrics_tf_only   = pd.read_csv(tf_metrics_path)
    plot_regime_comparison(metrics_sbs, metrics_tf_only, metrics_mrna_only, "phase4")
else:
    print("   ⚠️  Phase 3 metrics not found. Run Phase 3 notebook first.")
"""))

# ============================================================
# CELL 12 — Classification Report
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## 11. Detailed Classification Report (Best Model: XGBoost)"))
cells.append(nbf.v4.new_code_cell("""\
xgb_preds = fitted_sbs["XGBoost"].predict(X_te)
print("📋 Detailed Classification Report — XGBoost:")
print(classification_report(y_te, xgb_preds, target_names=["GS (0)", "CIN (1)"], digits=3))

# Show per-model summary
print("\\n📊 Final Test-Set Performance Summary (all models):")
summary_cols = ["Model", "10CV_ROC_AUC", "10CV_ROC_AUC_std", "Test_ROC_AUC",
                "Test_PR_AUC", "Test_Bal_Acc", "Test_F1"]
display(metrics_sbs[summary_cols].style
    .highlight_max(axis=0, subset=["Test_ROC_AUC","Test_PR_AUC","Test_F1"], color="#d4f1c4")
    .format(precision=4))
"""))

# ============================================================
# CELL 13 — Results Summary
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""\
## 12. Results Summary

### Key Findings

| Finding | Detail |
|---------|--------|
| **Best model (AUC)** | Random Forest (Test AUC = 0.975) |
| **Best SHAP feature** | `TF_GLI1` (Hedgehog pathway TF, mean SHAP = 0.957) |
| **Top mRNA feature** | `mRNA_AURKA` (Aurora Kinase A, mean SHAP = 0.608) |
| **Top SBS feature** | `MutSig_SBS2_13_APOBEC` (APOBEC signature, mean SHAP = 0.314) |
| **Significant SBS signatures** | SBS3 (HRD, p=0.015), SBS17 (Gastric/5-FU, p=0.009), **APOBEC (p=1.4×10⁻⁷)** |
| **SBS uplift** | +SBS adds minimal AUC gain over mRNA+TF alone, but confirms biological validity |

### Biological Interpretation

- **TF_GLI1** (top SHAP feature): GLI1 is a Hedgehog pathway transcription factor.  
  Its dominance in the XGBoost model suggests Hedgehog signalling differences between GS and CIN.
  
- **APOBEC enrichment in CIN** (SBS2/13, Δ mean = +0.116, p=1.4×10⁻⁷):  
  CIN tumours carry significantly higher APOBEC mutational burden, consistent with genomic instability.

- **HRD depletion in CIN** (SBS3, Δ mean = −0.071, p=0.015):  
  GS tumours show higher homologous recombination deficiency signature — consistent with their 
  EBV/MSI-enriched biology.

### Output Files
| File | Location |
|------|----------|
| `ml_mrna_tf_sbs_metrics.csv` | `results/` |
| `shap_feature_importances.csv` | `results/` |
| `sbs_differential_statistics.csv` | `results/` |
| `confusion_matrix_detail.csv` | `results/` |
| `phase4_shap_importances.png/.pdf` | `figures/` |
| `phase4_roc_pr.png/.pdf` | `figures/` |
| `phase4_confusion_matrices.png/.pdf` | `figures/` |
| `phase4_sbs_distributions.png/.pdf` | `figures/` |
| `phase4_regime_comparison.png/.pdf` | `figures/` |
| `mrna_tf_sbs_feature_matrix.csv` | `data/` |
"""))

# ============================================================
# CELL 14 — Load precomputed results (quick view)
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""\
## Appendix: Load Pre-computed Results

If you've already run the full pipeline (`run_phase4_rna_tf_mutsig.py`), 
use this cell to quickly load and inspect the saved results.
"""))
cells.append(nbf.v4.new_code_cell("""\
# Load and display pre-computed SHAP importances
shap_csv = os.path.join(RESULTS_DIR, "shap_feature_importances.csv")
if os.path.exists(shap_csv):
    df_shap = pd.read_csv(shap_csv)
    print("✅ Loaded shap_feature_importances.csv")
    print(f"   {len(df_shap)} features")
    print("\\nTop 20:")
    display(df_shap.head(20).style.background_gradient(subset=["Mean_SHAP"], cmap="YlOrRd"))
else:
    print("⚠️  shap_feature_importances.csv not found. Run cells above first.")

# Load ML metrics
metrics_csv = os.path.join(RESULTS_DIR, "ml_mrna_tf_sbs_metrics.csv")
if os.path.exists(metrics_csv):
    df_metrics = pd.read_csv(metrics_csv)
    print("\\n✅ Loaded ml_mrna_tf_sbs_metrics.csv")
    display(df_metrics[["Model","10CV_ROC_AUC","Test_ROC_AUC","Test_F1"]].style.format(precision=3))
"""))

# ============================================================
# Assemble notebook
# ============================================================
nb.cells = cells
out_path = "04_phase4_rna_tf_mutsig_classifier.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)

print(f"✅ Notebook written: {os.path.abspath(out_path)}")
