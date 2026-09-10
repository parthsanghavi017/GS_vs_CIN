import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BASE_DIR = "/media/parth/Research_Volume/GS_vs_CIN"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
STAD_PUB_DIR = os.path.join(BASE_DIR, "stad_tcga_pub")

def compute_fraction_genome_altered(seg_path, meta_df, cutoff=0.2):
    """
    Compute Fraction Genome Altered (FGA) per sample from segmented copy number data.
    FGA = sum(length of segments with |seg.mean| > cutoff) / total genome length in seg file
    """
    seg_df = pd.read_csv(seg_path, sep="\t")
    # Clean ID to 12-char patient barcode (e.g., TCGA-B7-5816-01 -> TCGA-B7-5816)
    seg_df["patient_barcode"] = seg_df["ID"].str[:12]
    seg_df["length"] = seg_df["loc.end"] - seg_df["loc.start"]
    
    # Filter for matched cohort patients
    common_patients = set(meta_df.index).intersection(set(seg_df["patient_barcode"]))
    seg_filtered = seg_df[seg_df["patient_barcode"].isin(common_patients)].copy()
    
    fga_records = []
    for patient, group in seg_filtered.groupby("patient_barcode"):
        total_length = group["length"].sum()
        altered_length = group.loc[group["seg.mean"].abs() > cutoff, "length"].sum()
        fga = altered_length / total_length if total_length > 0 else 0.0
        
        fga_records.append({
            "patient_barcode": patient,
            "total_bases": total_length,
            "altered_bases": altered_length,
            "FGA": fga,
            "MOLECULAR_SUBTYPE": meta_df.loc[patient, "MOLECULAR_SUBTYPE"]
        })
        
    fga_df = pd.DataFrame(fga_records).set_index("patient_barcode")
    out_path = os.path.join(RESULTS_DIR, "cnv_fga_burden.csv")
    fga_df.to_csv(out_path)
    print(f"Computed FGA for {len(fga_df)} samples. Results saved to: {out_path}")
    return fga_df

def profile_focal_cnv(cna_path, meta_df):
    """
    Profile high-level amplifications (+2) and homozygous deletions (-2) across driver genes.
    """
    cna_df = pd.read_csv(cna_path, sep="\t")
    cna_df = cna_df.drop_duplicates(subset=["Hugo_Symbol"]).set_index("Hugo_Symbol")
    
    # Drop annotation columns
    sample_cols = [c for c in cna_df.columns if c.startswith("TCGA-")]
    cna_matrix = cna_df[sample_cols].copy()
    
    # Rename columns to 12-char patient barcode
    patient_map = {c: c[:12] for c in sample_cols}
    cna_matrix = cna_matrix.rename(columns=patient_map)
    # Deduplicate columns if multiple samples exist per patient
    cna_matrix = cna_matrix.T.groupby(level=0).first().T
    
    common_patients = [p for p in meta_df.index if p in cna_matrix.columns]
    cna_matrix = cna_matrix[common_patients]
    subtypes = meta_df.loc[common_patients, "MOLECULAR_SUBTYPE"]
    
    driver_genes = [
        "ERBB2", "CCNE1", "CDK6", "EGFR", "MET", "FGFR2", "MYC", "KRAS", 
        "TP53", "CDKN2A", "CDH1", "ARID1A", "CLDN18", "RHOA", "CCND1", "MDM2"
    ]
    present_drivers = [g for g in driver_genes if g in cna_matrix.index]
    
    cin_mask = (subtypes == "CIN")
    gs_mask = (subtypes == "GS")
    
    driver_stats = []
    for gene in present_drivers:
        cin_series = cna_matrix.loc[gene, cin_mask]
        gs_series = cna_matrix.loc[gene, gs_mask]
        
        cin_amp_pct = (cin_series == 2).mean() * 100
        gs_amp_pct = (gs_series == 2).mean() * 100
        
        cin_del_pct = (cin_series == -2).mean() * 100
        gs_del_pct = (gs_series == -2).mean() * 100
        
        # Contingency for high amp (+2 vs other)
        table_amp = [[(cin_series == 2).sum(), (cin_series != 2).sum()],
                     [(gs_series == 2).sum(), (gs_series != 2).sum()]]
        _, pval_amp = stats.fisher_exact(table_amp)
        
        driver_stats.append({
            "Gene": gene,
            "CIN_Amp_Pct": cin_amp_pct,
            "GS_Amp_Pct": gs_amp_pct,
            "CIN_Del_Pct": cin_del_pct,
            "GS_Del_Pct": gs_del_pct,
            "Pval_Amp": pval_amp
        })
        
    focal_df = pd.DataFrame(driver_stats).sort_values("Pval_Amp", ascending=True)
    out_focal = os.path.join(RESULTS_DIR, "cnv_focal_alterations.csv")
    focal_df.to_csv(out_focal, index=False)
    print(f"Profiled focal driver CNVs. Results saved to: {out_focal}")
    return focal_df

def plot_cnv_figures(fga_df, focal_df):
    """Generate publication-quality CNV FGA boxplot and focal driver barplot."""
    # 1. FGA Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    sns.boxplot(
        data=fga_df,
        x="MOLECULAR_SUBTYPE",
        y="FGA",
        palette={"CIN": "#D95F02", "GS": "#1B9E77"},
        width=0.4,
        boxprops=dict(alpha=0.85),
        ax=ax
    )
    sns.stripplot(
        data=fga_df,
        x="MOLECULAR_SUBTYPE",
        y="FGA",
        color="black",
        alpha=0.5,
        jitter=0.2,
        size=5,
        ax=ax
    )
    
    cin_fga = fga_df.loc[fga_df["MOLECULAR_SUBTYPE"] == "CIN", "FGA"]
    gs_fga = fga_df.loc[fga_df["MOLECULAR_SUBTYPE"] == "GS", "FGA"]
    stat, pval = stats.mannwhitneyu(cin_fga, gs_fga)
    
    ax.set_title("Fraction Genome Altered (FGA): CIN vs. GS", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Molecular Subtype", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fraction Genome Altered (FGA)", fontsize=12, fontweight="bold")
    
    ax.text(
        0.5, 0.92,
        f"Mann-Whitney U $p = {pval:.2e}$",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
    )
    sns.despine()
    
    out_fga_fig = os.path.join(FIGURES_DIR, "cnv_fga_distribution.png")
    plt.tight_layout()
    plt.savefig(out_fga_fig, dpi=300)
    plt.close()
    print(f"FGA Distribution figure saved to: {out_fga_fig}")
    
    # 2. Focal Driver Alterations Plot
    plot_df = focal_df.melt(
        id_vars=["Gene"],
        value_vars=["CIN_Amp_Pct", "GS_Amp_Pct"],
        var_name="Subtype",
        value_name="Amplification_Pct"
    )
    plot_df["Subtype"] = plot_df["Subtype"].map({"CIN_Amp_Pct": "CIN", "GS_Amp_Pct": "GS"})
    
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    sns.barplot(
        data=plot_df,
        x="Gene",
        y="Amplification_Pct",
        hue="Subtype",
        palette={"CIN": "#D95F02", "GS": "#1B9E77"},
        ax=ax
    )
    
    ax.set_title("Focal High-Level Driver Amplification Frequencies (+2)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Driver Gene", fontsize=12, fontweight="bold")
    ax.set_ylabel("High-Level Amplification (%)", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontweight="bold")
    ax.legend(title="Molecular Subtype", frameon=True)
    sns.despine()
    
    out_driver_fig = os.path.join(FIGURES_DIR, "cnv_alteration_frequencies.png")
    plt.tight_layout()
    plt.savefig(out_driver_fig, dpi=300)
    plt.close()
    print(f"Driver CNV figure saved to: {out_driver_fig}")

def main():
    meta_path = os.path.join(RESULTS_DIR, "gs_cin_sample_metadata.csv")
    seg_path = os.path.join(STAD_PUB_DIR, "data_cna_hg19.seg")
    cna_path = os.path.join(STAD_PUB_DIR, "data_cna.txt")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}.")
        
    meta_df = pd.read_csv(meta_path, index_col=0)
    
    print("Step 1: Computing Fraction Genome Altered (FGA) per sample...")
    fga_df = compute_fraction_genome_altered(seg_path, meta_df)
    
    print("\nStep 2: Profiling focal driver CNV amplifications & deletions...")
    focal_df = profile_focal_cnv(cna_path, meta_df)
    
    print("\nStep 3: Generating CNV figures...")
    plot_cnv_figures(fga_df, focal_df)
    
    print("\nCNV profiling pipeline executed successfully!")

if __name__ == "__main__":
    main()
