import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import decoupler as dc

BASE_DIR = "/media/parth/Research_Volume/GS_vs_CIN"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

def compute_cpm_log1p(counts_df):
    """Compute Log2(CPM + 1) for decoupler TF activity inference."""
    lib_sizes = counts_df.sum(axis=0)
    cpm = (counts_df / lib_sizes) * 1e6
    log_cpm = np.log2(cpm + 1.0)
    return log_cpm

def get_dorothea_regulons(levels=['A', 'B', 'C']):
    """Fetch human DoRothEA network via decoupler."""
    print(f"Fetching DoRothEA human regulons for confidence levels: {levels}...")
    try:
        net = dc.op.dorothea(organism="human", levels=levels)
    except Exception as e:
        print(f"Fallback to CollectRI if DoRothEA encounters API changes: {e}")
        net = dc.op.collectri(organism="human")
    print(f"Retrieved regulon network with {len(net)} target interactions across {net['source'].nunique()} TFs.")
    return net

def infer_tf_activities(counts_df, meta_df, net):
    """Run decoupleR ULM (Univariate Linear Model) method to calculate sample-level TF activity scores."""
    log_cpm = compute_cpm_log1p(counts_df)
    
    # Transpose to (samples x genes) dataframe
    expr_matrix = log_cpm.T
    
    print("Running decoupleR ULM for TF activity inference...")
    tf_acts, tf_pvals = dc.mt.ulm(data=expr_matrix, net=net)
    
    out_tf_path = os.path.join(RESULTS_DIR, "tf_activity_scores.csv")
    tf_acts.to_csv(out_tf_path)
    print(f"TF activity score matrix saved to: {out_tf_path} with shape {tf_acts.shape} (samples x TFs)")
    
    return tf_acts, expr_matrix

def differential_tf_analysis(tf_acts, meta_df):
    """Perform differential TF activity analysis contrasting CIN vs GS."""
    common_samples = tf_acts.index.intersection(meta_df.index)
    tf_df = tf_acts.loc[common_samples]
    subtypes = meta_df.loc[common_samples, "MOLECULAR_SUBTYPE"]
    
    cin_mask = (subtypes == "CIN")
    gs_mask = (subtypes == "GS")
    
    diff_results = []
    
    for tf in tf_df.columns:
        cin_vals = tf_df.loc[cin_mask, tf].dropna()
        gs_vals = tf_df.loc[gs_mask, tf].dropna()
        
        if len(cin_vals) < 3 or len(gs_vals) < 3:
            continue
            
        stat, pval = stats.ttest_ind(cin_vals, gs_vals, equal_var=False)
        mean_cin = cin_vals.mean()
        mean_gs = gs_vals.mean()
        diff = mean_cin - mean_gs
        
        diff_results.append({
            "TF": tf,
            "mean_CIN": mean_cin,
            "mean_GS": mean_gs,
            "diff_CIN_vs_GS": diff,
            "stat": stat,
            "pvalue": pval
        })
        
    diff_df = pd.DataFrame(diff_results).sort_values("pvalue", ascending=True)
    
    # Benjamini-Hochberg FDR correction
    pvals = diff_df["pvalue"].values
    n = len(pvals)
    ranks = np.arange(1, n + 1)
    fdr = pvals * n / ranks
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    diff_df["padj"] = np.clip(fdr, 0, 1.0)
    
    out_diff_path = os.path.join(RESULTS_DIR, "differential_tf_activity_gs_vs_cin.csv")
    diff_df.to_csv(out_diff_path, index=False)
    print(f"Differential TF activity analysis complete. Saved to: {out_diff_path}")
    
    return diff_df

def plot_tf_figures(tf_acts, diff_df, meta_df, top_n=25):
    """Generate TF activity heatmap and volcano/rank plot."""
    # 1. TF Activity Heatmap
    top_tfs = diff_df.head(top_n)["TF"].values
    common_samples = tf_acts.index.intersection(meta_df.index)
    plot_matrix = tf_acts.loc[common_samples, top_tfs].T
    
    # Create colormap & sample metadata bar
    subtypes = meta_df.loc[common_samples, "MOLECULAR_SUBTYPE"]
    lut = {"CIN": "#D95F02", "GS": "#1B9E77"}
    col_colors = subtypes.map(lut)
    
    g = sns.clustermap(
        plot_matrix,
        col_colors=col_colors,
        cmap="vlag",
        center=0,
        z_score=0,
        figsize=(12, 8),
        cbar_kws={"label": "TF Activity Z-score"},
        dendrogram_ratio=(0.15, 0.15),
        linewidths=0.0
    )
    g.fig.suptitle("Top Differential Transcription Factor Activities: GS vs. CIN", y=1.02, fontsize=14, fontweight="bold")
    
    out_heatmap = os.path.join(FIGURES_DIR, "tf_activity_heatmap.png")
    g.savefig(out_heatmap, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"TF Activity Heatmap saved to: {out_heatmap}")
    
    # 2. Differential TF Volcano Plot
    df = diff_df.copy()
    df["neg_log10_padj"] = -np.log10(df["padj"].replace(0, 1e-300))
    df["Significance"] = "Not Significant"
    df.loc[(df["diff_CIN_vs_GS"] > 0) & (df["padj"] <= 0.05), "Significance"] = "Active in CIN"
    df.loc[(df["diff_CIN_vs_GS"] < 0) & (df["padj"] <= 0.05), "Significance"] = "Active in GS"
    
    palette = {"Active in CIN": "#D95F02", "Active in GS": "#1B9E77", "Not Significant": "#B0B0B0"}
    
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    sns.scatterplot(
        data=df,
        x="diff_CIN_vs_GS",
        y="neg_log10_padj",
        hue="Significance",
        palette=palette,
        s=50,
        alpha=0.85,
        ax=ax
    )
    
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.8)
    
    # Annotate top active TFs
    top_cin_tfs = df[df["Significance"] == "Active in CIN"].head(10)
    top_gs_tfs = df[df["Significance"] == "Active in GS"].head(10)
    annot_tfs = pd.concat([top_cin_tfs, top_gs_tfs])
    
    for idx, row in annot_tfs.iterrows():
        ax.annotate(
            row["TF"],
            (row["diff_CIN_vs_GS"], row["neg_log10_padj"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold"
        )
        
    ax.set_title("Differential TF Regulatory Activity: CIN vs. GS", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Mean TF Activity Difference [CIN - GS]", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$-\log_{10}(\text{Adjusted } p\text{-value})$", fontsize=12, fontweight="bold")
    ax.legend(title="Regulatory Activity", frameon=True)
    sns.despine()
    
    out_volcano = os.path.join(FIGURES_DIR, "tf_activity_volcano.png")
    plt.tight_layout()
    plt.savefig(out_volcano, dpi=300)
    plt.close()
    print(f"TF Activity Volcano plot saved to: {out_volcano}")

def main():
    counts_path = os.path.join(DATA_DIR, "tcga_stad_unstranded_counts.csv")
    meta_path = os.path.join(RESULTS_DIR, "gs_cin_sample_metadata.csv")
    
    if not os.path.exists(counts_path):
        raise FileNotFoundError(f"Count matrix not found at {counts_path}. Run download_tcga_counts.py first.")
        
    counts_df = pd.read_csv(counts_path, index_col=0)
    meta_df = pd.read_csv(meta_path, index_col=0)
    
    print("Step 1: Fetching DoRothEA regulons...")
    net = get_dorothea_regulons(levels=['A', 'B', 'C'])
    
    print("\nStep 2: Inferring TF activity scores per sample...")
    tf_acts, expr_matrix = infer_tf_activities(counts_df, meta_df, net)
    
    print("\nStep 3: Conducting Differential TF Activity Analysis...")
    diff_df = differential_tf_analysis(tf_acts, meta_df)
    
    print("\nStep 4: Generating TF Activity heatmaps and volcano plots...")
    plot_tf_figures(tf_acts, diff_df, meta_df)
    
    print("\ndecoupleR + DoRothEA TF activity pipeline executed successfully!")

if __name__ == "__main__":
    main()
