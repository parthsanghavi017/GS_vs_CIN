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

def profile_structural_variants(sv_path, meta_df):
    """
    Ingest data_sv.txt and build sample-level binary matrix for SV events.
    """
    sv_df = pd.read_csv(sv_path, sep="\t")
    sv_df["patient_barcode"] = sv_df["Sample_Id"].str[:12]
    
    common_patients = [p for p in meta_df.index if p in set(sv_df["patient_barcode"])]
    sv_filtered = sv_df[sv_df["patient_barcode"].isin(common_patients)].copy()
    
    # Create standardized event labels (e.g. SV_CLDN18_Fusion, SV_MYLK_Fusion, SV_CDKN2A_Silenced)
    sv_filtered["event_label"] = "SV_" + sv_filtered["Site1_Hugo_Symbol"] + "_" + sv_filtered["Event_Info"]
    
    # Unique events
    events = sorted(sv_filtered["event_label"].unique())
    
    # Binary matrix for all 190 cohort samples
    sv_matrix = pd.DataFrame(0, index=meta_df.index, columns=events)
    
    for _, row in sv_filtered.iterrows():
        patient = row["patient_barcode"]
        event = row["event_label"]
        if patient in sv_matrix.index and event in sv_matrix.columns:
            sv_matrix.loc[patient, event] = 1
            
    out_matrix = os.path.join(RESULTS_DIR, "sv_events_matrix.csv")
    sv_matrix.to_csv(out_matrix)
    print(f"Constructed binary SV matrix shape {sv_matrix.shape}. Saved to: {out_matrix}")
    
    # Differential SV Event Frequency Testing (CIN vs GS)
    subtypes = meta_df.loc[sv_matrix.index, "MOLECULAR_SUBTYPE"]
    cin_mask = (subtypes == "CIN")
    gs_mask = (subtypes == "GS")
    
    n_cin = cin_mask.sum()
    n_gs = gs_mask.sum()
    
    summary_records = []
    for event in events:
        cin_count = sv_matrix.loc[cin_mask, event].sum()
        gs_count = sv_matrix.loc[gs_mask, event].sum()
        
        cin_pct = (cin_count / n_cin) * 100
        gs_pct = (gs_count / n_gs) * 100
        
        table = [[cin_count, n_cin - cin_count],
                 [gs_count, n_gs - gs_count]]
        odds_ratio, pval = stats.fisher_exact(table)
        
        summary_records.append({
            "SV_Event": event,
            "CIN_Count": cin_count,
            "CIN_Pct": cin_pct,
            "GS_Count": gs_count,
            "GS_Pct": gs_pct,
            "Odds_Ratio": odds_ratio,
            "Fisher_Pval": pval
        })
        
    summary_df = pd.DataFrame(summary_records).sort_values("Fisher_Pval", ascending=True)
    out_summary = os.path.join(RESULTS_DIR, "structural_variant_summary.csv")
    summary_df.to_csv(out_summary, index=False)
    print(f"Differential SV testing complete. Saved to: {out_summary}")
    
    return sv_matrix, summary_df

def plot_sv_figures(summary_df):
    """Generate publication figure for SV event frequencies in CIN vs GS."""
    plot_df = summary_df.copy()
    plot_df["Event_Clean"] = plot_df["SV_Event"].str.replace("SV_", "").str.replace("_", " ")
    
    melted = plot_df.melt(
        id_vars=["Event_Clean"],
        value_vars=["CIN_Pct", "GS_Pct"],
        var_name="Subtype",
        value_name="Frequency_Pct"
    )
    melted["Subtype"] = melted["Subtype"].map({"CIN_Pct": "CIN", "GS_Pct": "GS"})
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.barplot(
        data=melted,
        x="Event_Clean",
        y="Frequency_Pct",
        hue="Subtype",
        palette={"CIN": "#D95F02", "GS": "#1B9E77"},
        ax=ax
    )
    
    ax.set_title("Structural Variant & Gene Fusion Frequencies: CIN vs. GS", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Structural Variant Event", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cohort Frequency (%)", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontweight="bold")
    ax.legend(title="Molecular Subtype", frameon=True)
    sns.despine()
    
    out_fig = os.path.join(FIGURES_DIR, "structural_variants_distribution.png")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Structural Variants figure saved to: {out_fig}")

def main():
    meta_path = os.path.join(RESULTS_DIR, "gs_cin_sample_metadata.csv")
    sv_path = os.path.join(STAD_PUB_DIR, "data_sv.txt")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}.")
        
    meta_df = pd.read_csv(meta_path, index_col=0)
    
    print("Step 1: Profiling Structural Variants and Fusions from data_sv.txt...")
    sv_matrix, summary_df = profile_structural_variants(sv_path, meta_df)
    
    print("\nStep 2: Generating Structural Variant distribution plot...")
    plot_sv_figures(summary_df)
    
    print("\nStructural Variants pipeline executed successfully!")

if __name__ == "__main__":
    main()
