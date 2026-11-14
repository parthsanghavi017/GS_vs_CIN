#!/usr/bin/env python3
"""
Phase 4: mRNA + Transcription Factor Activity + COSMIC Mutational Signatures
=============================================================================
TCGA-STAD: Genomically Stable (GS) vs. Chromosomal Instability (CIN)

Experiment: Can we classify GS vs. CIN using only:
  - 46 DESeq2 significant mRNA transcripts
  - 30 DoRothEA transcription factor regulon activities
  - 5 COSMIC SBS mutational signatures (SBS1, SBS3, SBS5, SBS17, SBS2_13_APOBEC)
  = 81 total features, ZERO CNA / ZERO CNV

Author: GS_vs_CIN-main analysis pipeline
"""

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
    f1_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
import shap

# ===========================================================================
# AESTHETICS
# ===========================================================================
sns.set_theme(style="ticks", font="sans-serif")
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.5
})

PALETTE = {"GS": "#2b7bba", "CIN": "#d62728"}
MODEL_COLORS = {
    "Random Forest":      "#1f77b4",
    "XGBoost":            "#ff7f0e",
    "LightGBM":           "#2ca02c",
    "Logistic Regression":"#9467bd",
    "Support Vector Machine": "#d62728"
}

# ===========================================================================
# PATHS
# ===========================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
FIGURES_DIR  = os.path.join(BASE_DIR, "figures")
MAIN_RESULTS = os.path.join(BASE_DIR, "..", "results")
PHASE3_DATA  = os.path.join(BASE_DIR, "..", "phase3_rna_only", "data")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ===========================================================================
# FEATURE DEFINITIONS
# ===========================================================================
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


# ===========================================================================
# HELPER: ML EVALUATION
# ===========================================================================
def evaluate_ml_pipelines(X_train_df, X_test_df, y_tr, y_te, exp_name="mRNA+TF+SBS"):
    """Train 5 classifiers with SMOTE+10-fold CV and evaluate on holdout test set."""
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


# ===========================================================================
# PLOT: ROC + PR CURVES
# ===========================================================================
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
    ax1.set_title(f"ROC Curves\n{title_prefix}", fontweight="bold", fontsize=12)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.legend(fontsize=8, loc="lower right"); ax1.grid(True, alpha=0.3)

    base_prec = y_test.sum() / len(y_test)
    ax2.axhline(base_prec, color="gray", linestyle="--", lw=1, label=f"Baseline ({base_prec:.2f})")
    ax2.set_title(f"Precision-Recall Curves\n{title_prefix}", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.legend(fontsize=8, loc="lower left"); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_roc_pr.png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_roc_pr.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved ROC/PR curves → {save_prefix}_roc_pr.png")


# ===========================================================================
# PLOT: CONFUSION MATRICES
# ===========================================================================
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
        ax.set_title(f"{name}\nAcc={((tp+tn)/len(y_test)):.1%}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

        detail_rows.append({
            "Model": name,
            "TN_GS_Correct": int(tn), "FP_GS_Called_CIN": int(fp),
            "FN_CIN_Called_GS": int(fn), "TP_CIN_Correct": int(tp),
            "Sensitivity_CIN": f"{tp/(tp+fn):.1%}",
            "Specificity_GS":  f"{tn/(tn+fp):.1%}",
            "Overall_Accuracy": f"{(tp+tn)/len(y_test):.1%}"
        })

    plt.suptitle(f"Confusion Matrices — {title_prefix}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_confusion_matrices.png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_confusion_matrices.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved confusion matrices → {save_prefix}_confusion_matrices.png")
    return pd.DataFrame(detail_rows)


# ===========================================================================
# PLOT: SHAP FEATURE IMPORTANCE
# ===========================================================================
def plot_shap_importances(pipe, X_train, X_test, feature_names, save_prefix, top_n=25):
    try:
        clf = pipe.named_steps["clf"]
        scaler = pipe.named_steps["scaler"]
        X_te_scaled = scaler.transform(X_test).astype(np.float32)

        # Use XGBoost native SHAP (pred_contribs) — works with XGBoost 3.x
        # The SHAP library has a known incompatibility with XGBoost 3.x JSON format.
        from xgboost import DMatrix
        booster = clf.get_booster()
        dmat = DMatrix(pd.DataFrame(X_te_scaled, columns=feature_names))
        # pred_contribs returns shape (n_samples, n_features + 1); last col = bias
        contrib_matrix = booster.predict(dmat, pred_contribs=True)
        vals = contrib_matrix[:, :-1]          # drop bias column

        mean_abs = np.abs(vals).mean(axis=0)
        shap_df  = pd.DataFrame({"Feature": feature_names, "Mean_SHAP": mean_abs})
        shap_df    = shap_df.sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)

        # Colour by modality
        def _modality_color(feat):
            if feat.startswith("MutSig_"):  return "#e31a1c"
            elif feat.startswith("mRNA_") or feat in [g for g in MRNA_GENES]: return "#1f78b4"
            else:                            return "#33a02c"

        top = shap_df.head(top_n)
        colors = [_modality_color(f) for f in top["Feature"]]

        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        bars = ax.barh(top["Feature"][::-1], top["Mean_SHAP"][::-1], color=colors[::-1], edgecolor="white", height=0.7)
        ax.set_xlabel("Mean |SHAP Value|  (impact on CIN prediction)", fontsize=11)
        ax.set_title(f"Top {top_n} Feature Importances (SHAP)\nmRNA + TF + COSMIC SBS", fontweight="bold", fontsize=13)
        ax.grid(True, axis="x", alpha=0.3)

        # Legend for modality colours
        legend_patches = [
            mpatches.Patch(color="#1f78b4", label="mRNA Gene Expression"),
            mpatches.Patch(color="#33a02c", label="TF Regulon Activity (DoRothEA)"),
            mpatches.Patch(color="#e31a1c", label="COSMIC SBS Mutational Signature"),
        ]
        ax.legend(handles=legend_patches, fontsize=9, loc="lower right")

        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_shap_importances.png"), bbox_inches="tight", dpi=200)
        fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_shap_importances.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 Saved SHAP importance plot → {save_prefix}_shap_importances.png")
        return shap_df
    except Exception as e:
        print(f"  ⚠️  SHAP computation failed: {e}")
        return pd.DataFrame()


# ===========================================================================
# PLOT: SHAP BEESWARM (signed values — direction reveals GS vs CIN drivers)
# ===========================================================================
def plot_shap_beeswarm(pipe, X_test, feature_names, save_prefix, top_n=25):
    """
    Beeswarm plot of raw (signed) SHAP values.
    Each dot = one patient.
    X-axis = SHAP value:  positive → pushes toward CIN prediction
                          negative → pushes toward GS prediction
    Dot colour = actual feature value (red=high, blue=low).
    Features ranked by mean |SHAP| (most important at top).
    """
    try:
        clf    = pipe.named_steps["clf"]
        scaler = pipe.named_steps["scaler"]
        X_te_scaled = scaler.transform(X_test).astype(np.float32)

        # Signed SHAP via XGBoost native pred_contribs
        from xgboost import DMatrix
        booster = clf.get_booster()
        dmat    = DMatrix(pd.DataFrame(X_te_scaled, columns=feature_names))
        contrib_matrix = booster.predict(dmat, pred_contribs=True)
        vals    = contrib_matrix[:, :-1].astype(np.float32)   # drop bias
        base_val = contrib_matrix[:, -1].mean()               # mean bias

        # Build shap.Explanation so we can use shap.plots.beeswarm
        explanation = shap.Explanation(
            values        = vals,
            base_values   = np.full(len(X_test), base_val),
            data          = X_te_scaled,
            feature_names = list(feature_names)
        )

        # Rank features by mean |SHAP| for ordering
        mean_abs_order = np.argsort(np.abs(vals).mean(axis=0))[::-1]
        top_idx        = mean_abs_order[:top_n]
        explanation_top = shap.Explanation(
            values        = vals[:, top_idx],
            base_values   = explanation.base_values,
            data          = X_te_scaled[:, top_idx],
            feature_names = [feature_names[i] for i in top_idx]
        )

        fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
        shap.plots.beeswarm(explanation_top, max_display=top_n, show=False,
                            color_bar=True)
        plt.title(f"SHAP Beeswarm — Top {top_n} Features (XGBoost)\n"
                  "mRNA + TF + COSMIC SBS  |  Positive = CIN driver, Negative = GS driver",
                  fontweight="bold", fontsize=12, pad=14)
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_shap_beeswarm.png"),
                    bbox_inches="tight", dpi=200)
        fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_shap_beeswarm.pdf"),
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 Saved SHAP beeswarm → {save_prefix}_shap_beeswarm.png")
    except Exception as e:
        print(f"  ⚠️  SHAP beeswarm failed: {e}")


# ===========================================================================
# PLOT: SBS SIGNATURE DISTRIBUTIONS (GS vs CIN Boxplots)
# ===========================================================================
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
        sns.boxplot(data=data_plot, x="Subtype", y="Exposure", palette=PALETTE, ax=ax,
                    order=["GS", "CIN"], width=0.5, linewidth=1.5)
        sns.stripplot(data=data_plot, x="Subtype", y="Exposure", palette=PALETTE, ax=ax,
                      order=["GS", "CIN"], size=3, alpha=0.5, jitter=True)

        display_name = SBS_DISPLAY_NAMES.get(col, col)
        ax.set_title(f"{display_name}\np = {p_val:.2e}", fontweight="bold", fontsize=10)
        ax.set_xlabel(""); ax.set_ylabel("Mutational Exposure")
        ax.grid(True, axis="y", alpha=0.3)

        stat_rows.append({
            "Signature": col,
            "CIN_Mean": cin_vals.mean(), "GS_Mean": gs_vals.mean(),
            "Delta_CIN_minus_GS": cin_vals.mean() - gs_vals.mean(),
            "MannWhitney_U": t_stat, "p_value": p_val,
            "Significant_p05": p_val < 0.05
        })

    plt.suptitle("COSMIC SBS Mutational Signature Exposures: GS vs. CIN", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_sbs_distributions.png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_sbs_distributions.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved SBS distributions → {save_prefix}_sbs_distributions.png")
    return pd.DataFrame(stat_rows)


# ===========================================================================
# PLOT: REGIME COMPARISON BAR CHART
# ===========================================================================
def plot_regime_comparison(metrics_sbs, metrics_tf_only, metrics_mrna_only, save_prefix):
    """Compare mRNA-only vs mRNA+TF vs mRNA+TF+SBS on Test ROC-AUC and Test F1."""
    combined = pd.concat([
        metrics_mrna_only.assign(Regime="Pure mRNA (46 genes)"),
        metrics_tf_only.assign(Regime="mRNA + TF (76 features)"),
        metrics_sbs.assign(Regime="mRNA + TF + SBS (81 features)"),
    ])

    palette_regime = {
        "Pure mRNA (46 genes)":     "#2b5c8f",
        "mRNA + TF (76 features)":  "#7570b3",
        "mRNA + TF + SBS (81 features)": "#e31a1c",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    sns.barplot(data=combined, x="Model", y="Test_ROC_AUC", hue="Regime",
                palette=palette_regime, ax=ax1)
    ax1.set_title("Holdout Test ROC-AUC\nacross RNA Feature Regimes", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Test ROC-AUC"); ax1.set_ylim(0.78, 1.0)
    ax1.grid(True, axis="y", alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")

    sns.barplot(data=combined, x="Model", y="Test_F1", hue="Regime",
                palette=palette_regime, ax=ax2)
    ax2.set_title("Holdout Test F1-Score\nacross RNA Feature Regimes", fontweight="bold", fontsize=12)
    ax2.set_ylabel("Test F1-Score"); ax2.set_ylim(0.78, 1.0)
    ax2.grid(True, axis="y", alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_regime_comparison.png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_regime_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved regime comparison → {save_prefix}_regime_comparison.png")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("🚀 PHASE 4: mRNA + TF + COSMIC SBS MUTATIONAL SIGNATURES CLASSIFIER")
    print("   Zero CNA / Zero CNV — Pure transcriptomics + mutagenesis")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. LOAD DATA FROM PHASE 2 INTEGRATED FEATURE STORE
    # -----------------------------------------------------------------------
    feature_store = os.path.join(MAIN_RESULTS, "phase2_integrated_feature_store.csv")
    print(f"\n📖 Loading feature store: {feature_store}")
    df_full = pd.read_csv(feature_store, index_col=0)
    print(f"   Loaded: {df_full.shape[0]} patients × {df_full.shape[1]} features")

    # Load TF feature matrix from phase3 to identify which TF columns are present
    mrna_tf_file = os.path.join(PHASE3_DATA, "mrna_tf_feature_matrix.csv")
    df_mrna_tf   = pd.read_csv(mrna_tf_file, index_col=0)
    mrna_tf_cols = [c for c in df_mrna_tf.columns if c != "MOLECULAR_SUBTYPE"]
    mrna_cols    = [c for c in mrna_tf_cols if c.startswith("mRNA_")]
    tf_cols_phase3 = [c for c in mrna_tf_cols if not c.startswith("mRNA_")]

    print(f"\n   mRNA columns from Phase 3 matrix: {len(mrna_cols)}")
    print(f"   TF  columns from Phase 3 matrix : {len(tf_cols_phase3)}")

    # Identify SBS columns in feature store
    sbs_cols_available = [c for c in df_full.columns if c.startswith("MutSig_")]
    print(f"   SBS columns in feature store   : {len(sbs_cols_available)} → {sbs_cols_available}")

    # Build combined dataset: align on index
    target_col = "SUBTYPE_CIN_VS_GS"
    label_col  = "MOLECULAR_SUBTYPE"

    # Normalize phase3 index FIRST (strip '-01' suffix) to match feature store format
    df_mrna_tf_norm = df_mrna_tf.copy()
    df_mrna_tf_norm.index = df_mrna_tf_norm.index.str.replace(r"-\d{2}$", "", regex=True)

    # Now filter to patients present in both datasets
    common_patients = df_mrna_tf_norm.index.intersection(df_full.index)
    print(f"   Common patients (overlap after index normalization): {len(common_patients)}")

    df_mrna_tf_aligned = df_mrna_tf_norm.loc[common_patients].copy()
    df_sbs_aligned     = df_full.loc[common_patients, sbs_cols_available + [label_col]].copy()

    # Merge
    df_combined = df_mrna_tf_aligned.join(df_sbs_aligned[sbs_cols_available], how="inner")
    print(f"\n✅ Combined dataset: {df_combined.shape[0]} patients × {df_combined.shape[1]} columns")

    # Add subtype string for plotting
    df_combined["SUBTYPE"] = df_sbs_aligned[label_col]

    # Features + target
    X_all = df_combined.drop(columns=["MOLECULAR_SUBTYPE", "SUBTYPE"])
    y_all = (df_combined["MOLECULAR_SUBTYPE"] == "CIN").astype(int)

    print(f"   Feature matrix: {X_all.shape[0]} × {X_all.shape[1]}")
    print(f"   CIN: {(y_all==1).sum()}, GS: {(y_all==0).sum()}")

    # Save the combined feature matrix for reference
    X_all.join(df_combined[["MOLECULAR_SUBTYPE"]]).to_csv(
        os.path.join(DATA_DIR, "mrna_tf_sbs_feature_matrix.csv"))
    print(f"   💾 Feature matrix saved → data/mrna_tf_sbs_feature_matrix.csv")

    # -----------------------------------------------------------------------
    # 2. TRAIN / TEST SPLIT (same seed as Phase 3 for fair comparison)
    # -----------------------------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=42
    )
    print(f"\n📊 Train: {len(X_tr)} | Test: {len(X_te)}")
    print(f"   Test  — CIN: {(y_te==1).sum()}, GS: {(y_te==0).sum()}")

    # -----------------------------------------------------------------------
    # 3. SBS DISTRIBUTION ANALYSIS (Before ML)
    # -----------------------------------------------------------------------
    print("\n📈 Computing SBS signature differential analysis...")
    sbs_stat_df = plot_sbs_distributions(df_combined, "phase4")
    sbs_stat_df.to_csv(os.path.join(RESULTS_DIR, "sbs_differential_statistics.csv"), index=False)
    print("   💾 SBS stats saved → results/sbs_differential_statistics.csv")
    print(sbs_stat_df[["Signature","CIN_Mean","GS_Mean","Delta_CIN_minus_GS","p_value","Significant_p05"]].to_string())

    # -----------------------------------------------------------------------
    # 4. MACHINE LEARNING BENCHMARK
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("🤖 RUNNING ML CLASSIFIERS: mRNA + TF + SBS (81 features)")
    print("=" * 70)
    metrics_sbs, fitted_sbs = evaluate_ml_pipelines(X_tr, X_te, y_tr, y_te, "mRNA + TF + SBS")

    # Save metrics
    metrics_sbs.to_csv(os.path.join(RESULTS_DIR, "ml_mrna_tf_sbs_metrics.csv"), index=False)
    print("\n🏆 mRNA + TF + SBS Results:")
    print(metrics_sbs[["Model","10CV_ROC_AUC","10CV_ROC_AUC_std","Test_ROC_AUC",
                         "Test_PR_AUC","Test_Bal_Acc","Test_F1"]].to_string())

    # -----------------------------------------------------------------------
    # 5. VISUALIZATIONS
    # -----------------------------------------------------------------------
    print("\n🎨 Generating figures...")
    plot_roc_pr(fitted_sbs, X_te, y_te, "mRNA + TF + COSMIC SBS (Zero CNA/CNV)", "phase4")
    cm_detail = plot_confusion_matrices(fitted_sbs, X_te, y_te, "mRNA + TF + COSMIC SBS", "phase4")
    cm_detail.to_csv(os.path.join(RESULTS_DIR, "confusion_matrix_detail.csv"), index=False)
    print("   Confusion matrix breakdown:")
    print(cm_detail.to_string(index=False))

    # SHAP on best CV model (XGBoost had highest 10-CV AUC = 0.915)
    print("\n🔍 Computing SHAP importances (XGBoost, Top 25)...")
    shap_df = plot_shap_importances(
        fitted_sbs["XGBoost"], X_tr, X_te, X_all.columns.tolist(), "phase4", top_n=25
    )
    plot_shap_importances(
        fitted_sbs["XGBoost"], X_tr, X_te, X_all.columns.tolist(), "phase4_top30", top_n=30
    )
    if not shap_df.empty:
        shap_df.to_csv(os.path.join(RESULTS_DIR, "shap_feature_importances.csv"), index=False)
        print("   💾 SHAP importances saved → results/shap_feature_importances.csv")
        print("\n   Top 15 Features by SHAP:")
        print(shap_df.head(15)[["Feature","Mean_SHAP"]].to_string(index=False))

    # SHAP beeswarm — shows direction (positive=CIN, negative=GS)
    print("\n🐝 Generating SHAP beeswarm plot (Top 25)...")
    plot_shap_beeswarm(
        fitted_sbs["XGBoost"], X_te, X_all.columns.tolist(), "phase4", top_n=25
    )
    plot_shap_beeswarm(
        fitted_sbs["XGBoost"], X_te, X_all.columns.tolist(), "phase4_top30", top_n=30
    )

    # -----------------------------------------------------------------------
    # 6. LOAD PHASE 3 METRICS FOR REGIME COMPARISON PLOT
    # -----------------------------------------------------------------------
    print("\n📊 Loading Phase 3 metrics for 3-way regime comparison...")
    mrna_metrics_path = os.path.join(BASE_DIR, "..", "phase3_rna_only", "results", "ml_pure_mrna_only_metrics.csv")
    tf_metrics_path   = os.path.join(BASE_DIR, "..", "phase3_rna_only", "results", "ml_mrna_tf_metrics.csv")

    if os.path.exists(mrna_metrics_path) and os.path.exists(tf_metrics_path):
        metrics_mrna_only = pd.read_csv(mrna_metrics_path)
        metrics_tf_only   = pd.read_csv(tf_metrics_path)
        plot_regime_comparison(metrics_sbs, metrics_tf_only, metrics_mrna_only, "phase4")
    else:
        print("   ⚠️  Phase 3 metrics not found, skipping regime comparison plot.")

    # -----------------------------------------------------------------------
    # 7. CLASSIFICATION REPORT (Detailed Per-Class Metrics)
    # -----------------------------------------------------------------------
    print("\n📋 Detailed Classification Reports (Best model: XGBoost):")
    xgb_preds = fitted_sbs["XGBoost"].predict(X_te)
    print(classification_report(y_te, xgb_preds, target_names=["GS (0)", "CIN (1)"],
                                 digits=3))

    print("\n" + "=" * 70)
    print("✅ PHASE 4 ANALYSIS COMPLETE!")
    print(f"   📁 Results  : {RESULTS_DIR}")
    print(f"   📁 Figures  : {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
