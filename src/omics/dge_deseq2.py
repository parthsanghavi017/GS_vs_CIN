import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import gseapy as gp

BASE_DIR = "/media/parth/Research_Volume/GS_vs_CIN"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

def run_deseq2(counts_path, meta_path):
    """Load unstranded count matrix and run PyDESeq2 contrasting CIN vs GS."""
    counts_df = pd.read_csv(counts_path, index_col=0)
    meta_df = pd.read_csv(meta_path, index_col=0)
    
    # Align samples
    common_samples = [s for s in meta_df.index if s in counts_df.columns]
    counts_df = counts_df[common_samples]
    meta_df = meta_df.loc[common_samples]
    
    # Filter out non-numeric or NaN values and low counts
    counts_df = counts_df.fillna(0).astype(int)
    # Remove genes with total count across samples < 10
    gene_sums = counts_df.sum(axis=1)
    counts_df = counts_df.loc[gene_sums >= 10]
    
    print(f"Preprocessed count matrix shape for DESeq2: {counts_df.shape[0]} genes x {counts_df.shape[1]} samples.")
    print("Sample metadata subtypes:\n", meta_df["MOLECULAR_SUBTYPE"].value_counts())
    
    # Transpose counts matrix (samples x genes) for PyDESeq2
    counts_transposed = counts_df.T
    
    dds = DeseqDataSet(
        counts=counts_transposed,
        metadata=meta_df,
        design_factors="MOLECULAR_SUBTYPE",
        ref_level=["MOLECULAR_SUBTYPE", "GS"],
        n_cpus=4,
        quiet=False
    )
    
    dds.deseq2()
    
    stat_res = DeseqStats(dds, contrast=["MOLECULAR_SUBTYPE", "CIN", "GS"], n_cpus=4)
    stat_res.summary()
    res_df = stat_res.results_df.copy()
    
    # Sort by padj / pvalue
    res_df = res_df.sort_values("padj", ascending=True)
    out_deg_path = os.path.join(RESULTS_DIR, "dge_deseq2_gs_vs_cin.csv")
    res_df.to_csv(out_deg_path)
    print(f"DESeq2 analysis complete. DEG results saved to: {out_deg_path}")
    
    return res_df, dds

def plot_volcano(res_df, top_n=15):
    """Generate publication-quality volcano plot for DESeq2 results."""
    df = res_df.copy().dropna(subset=["log2FoldChange", "pvalue", "padj"])
    df["neg_log10_padj"] = -np.log10(df["padj"].replace(0, 1e-300))
    
    # Define significance criteria
    fc_cutoff = 1.0
    pval_cutoff = 0.05
    
    df["Significance"] = "Not Significant"
    df.loc[(df["log2FoldChange"] >= fc_cutoff) & (df["padj"] <= pval_cutoff), "Significance"] = "Upregulated in CIN"
    df.loc[(df["log2FoldChange"] <= -fc_cutoff) & (df["padj"] <= pval_cutoff), "Significance"] = "Upregulated in GS"
    
    palette = {
        "Upregulated in CIN": "#D95F02",  # Bright orange/red
        "Upregulated in GS": "#1B9E77",   # Deep teal/green
        "Not Significant": "#B0B0B0"
    }
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.scatterplot(
        data=df,
        x="log2FoldChange",
        y="neg_log10_padj",
        hue="Significance",
        palette=palette,
        alpha=0.75,
        s=35,
        edgecolor=None,
        ax=ax
    )
    
    ax.axvline(x=fc_cutoff, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(x=-fc_cutoff, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(y=-np.log10(pval_cutoff), color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    
    # Annotate top genes
    top_cin = df[df["Significance"] == "Upregulated in CIN"].sort_values("padj").head(top_n)
    top_gs = df[df["Significance"] == "Upregulated in GS"].sort_values("padj").head(top_n)
    
    # Key hallmark gastric cancer genes to highlight if present
    hallmarks = ["CDH1", "RHOA", "CLDN18", "ERBB2", "CCNE1", "FGFR2", "EGFR", "TP53", "MET", "CDK6"]
    special_genes = df.loc[df.index.intersection(hallmarks)]
    
    annot_df = pd.concat([top_cin, top_gs, special_genes]).drop_duplicates()
    
    for gene, row in annot_df.iterrows():
        ax.annotate(
            gene,
            (row["log2FoldChange"], row["neg_log10_padj"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold" if gene in hallmarks else "normal",
            alpha=0.9
        )
        
    ax.set_title("DESeq2 Differential Expression: CIN vs. GS Gastric Cancer", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(r"$\log_2(\text{Fold Change})$ [CIN / GS]", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$-\log_{10}(\text{Adjusted } p\text{-value})$", fontsize=12, fontweight="bold")
    ax.legend(title="Subtype Enriched", frameon=True, facecolor="white", edgecolor="none")
    sns.despine()
    
    out_fig = os.path.join(FIGURES_DIR, "volcano_dge_deseq2.png")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Volcano plot saved to: {out_fig}")

def run_gsea(res_df):
    """Perform pre-ranked GSEA on DESeq2 results against MSigDB Hallmark gene sets."""
    df = res_df.dropna(subset=["stat"]).copy()
    # Rank by stat column (Wald statistic)
    rnk = df["stat"].sort_values(ascending=False)
    
    print("Running GSEA Pre-ranked against MSigDB Hallmark 2020...")
    pre_res = gp.prerank(
        rnk=rnk,
        gene_sets="MSigDB_Hallmark_2020",
        processes=4,
        permutation_num=1000,
        outdir=None,
        seed=42
    )
    
    gsea_df = pre_res.res2d.sort_values("FDR q-val", ascending=True).reset_index()
    out_gsea_path = os.path.join(RESULTS_DIR, "gsea_hallmark_results.csv")
    gsea_df.to_csv(out_gsea_path, index=False)
    print(f"GSEA Hallmark analysis complete. Results saved to: {out_gsea_path}")
    
    # Plot top enriched pathways
    top_pathways = gsea_df.head(20).copy()
    top_pathways["Term"] = top_pathways["Term"].str.replace("MSigDB_Hallmark_2020__", "").str.replace("_", " ")
    top_pathways["Enrichment"] = np.where(top_pathways["NES"] > 0, "CIN Enriched", "GS Enriched")
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    sns.barplot(
        data=top_pathways,
        x="NES",
        y="Term",
        hue="Enrichment",
        palette={"CIN Enriched": "#D95F02", "GS Enriched": "#1B9E77"},
        ax=ax
    )
    
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("GSEA Hallmark Pathways: CIN vs. GS", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Hallmark Pathway", fontsize=12, fontweight="bold")
    ax.legend(title="Enrichment Direction", frameon=True)
    sns.despine()
    
    out_gsea_fig = os.path.join(FIGURES_DIR, "gsea_hallmark_enr.png")
    plt.tight_layout()
    plt.savefig(out_gsea_fig, dpi=300)
    plt.close()
    print(f"GSEA enrichment plot saved to: {out_gsea_fig}")
    
    return gsea_df

def main():
    counts_path = os.path.join(DATA_DIR, "tcga_stad_unstranded_counts.csv")
    meta_path = os.path.join(RESULTS_DIR, "gs_cin_sample_metadata.csv")
    
    if not os.path.exists(counts_path):
        raise FileNotFoundError(f"Count matrix not found at {counts_path}. Please run download_tcga_counts.py first.")
        
    print("Step 1: Running PyDESeq2 DGE analysis...")
    res_df, dds = run_deseq2(counts_path, meta_path)
    
    print("\nStep 2: Generating Volcano Plot...")
    plot_volcano(res_df)
    
    print("\nStep 3: Running GSEA Hallmark Enrichment...")
    gsea_df = run_gsea(res_df)
    
    print("\nDESeq2 + GSEA pipeline executed successfully!")

if __name__ == "__main__":
    main()
