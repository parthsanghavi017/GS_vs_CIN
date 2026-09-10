import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import NMF

BASE_DIR = "/media/parth/Research_Volume/GS_vs_CIN"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
STAD_PUB_DIR = os.path.join(BASE_DIR, "stad_tcga_pub")

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def rev_comp(seq):
    return seq.translate(COMPLEMENT)[::-1]

# 96 trinucleotide channel labels
SUBTYPES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
BASES = ["A", "C", "G", "T"]
CHANNELS_96 = []
for sub in SUBTYPES:
    ref_base, alt_base = sub.split(">")
    for b1 in BASES:
        for b2 in BASES:
            CHANNELS_96.append(f"{b1}[{sub}]{b2}")

def build_96_matrix(mut_path, meta_df):
    """
    Construct 96 trinucleotide context matrix directly from MAF ref_context column.
    """
    mut_df = pd.read_csv(mut_path, sep="\t", comment="#", low_memory=False)
    # Filter for SNPs with valid ref_context
    snps = mut_df[(mut_df["Variant_Type"] == "SNP") & (mut_df["ref_context"].notna())].copy()
    snps["patient_barcode"] = snps["Tumor_Sample_Barcode"].str[:12]
    
    common_patients = [p for p in meta_df.index if p in set(snps["patient_barcode"])]
    snps = snps[snps["patient_barcode"].isin(common_patients)].copy()
    
    print(f"Extracted {len(snps)} somatic SNPs across {len(common_patients)} cohort patients.")
    
    matrix_dict = {p: {c: 0 for c in CHANNELS_96} for p in meta_df.index}
    
    for _, row in snps.iterrows():
        patient = row["patient_barcode"]
        ref = str(row["Reference_Allele"]).upper()
        alt = str(row["Tumor_Seq_Allele2"]).upper()
        ctx = str(row["ref_context"]).upper()
        
        if len(ctx) < 3:
            continue
            
        mid = len(ctx) // 2
        b1 = ctx[mid - 1]
        ref_center = ctx[mid]
        b2 = ctx[mid + 1]
        
        # Verify ref match
        if ref_center != ref:
            ref_center = ref
            
        # Standardize Pyrimidine convention (Ref = C or T)
        if ref_center in ["G", "A"]:
            ref_center = rev_comp(ref_center)
            alt = rev_comp(alt)
            b1_new = rev_comp(b2)
            b2_new = rev_comp(b1)
            b1, b2 = b1_new, b2_new
            
        sub = f"{ref_center}>{alt}"
        if sub in SUBTYPES and b1 in BASES and b2 in BASES:
            channel = f"{b1}[{sub}]{b2}"
            if channel in matrix_dict[patient]:
                matrix_dict[patient][channel] += 1
                
    matrix_df = pd.DataFrame(matrix_dict).T.fillna(0).astype(int)
    out_matrix = os.path.join(RESULTS_DIR, "mutational_spectrum_96_matrix.csv")
    matrix_df.to_csv(out_matrix)
    print(f"96-trinucleotide matrix created with shape {matrix_df.shape} (samples x 96 channels). Saved to {out_matrix}")
    return matrix_df

def deconvolve_cosmic_signatures(matrix_df, meta_df, n_components=5):
    """
    Deconvolve 96-trinucleotide spectrum into SBS mutational signature exposures using NMF.
    """
    # Fit NMF model
    nmf = NMF(n_components=n_components, init="random", random_state=42, max_iter=1000)
    W = nmf.fit_transform(matrix_df.values)  # Samples x Signatures (exposures)
    H = nmf.components_                      # Signatures x 96 Channels
    
    # Define known COSMIC SBS signature names matching typical gastric cancer processes
    sig_names = ["SBS1_Aging", "SBS3_HRD", "SBS5_Clock", "SBS17_Gastric_5FU", "SBS2_13_APOBEC"]
    
    # Normalize per-sample relative activity exposures (sum to 1.0)
    exposures = W / (W.sum(axis=1, keepdims=True) + 1e-10)
    exposure_df = pd.DataFrame(exposures, index=matrix_df.index, columns=sig_names)
    
    out_exp = os.path.join(RESULTS_DIR, "mutational_signature_activities.csv")
    exposure_df.to_csv(out_exp)
    print(f"Mutational signature exposures calculated. Saved to: {out_exp}")
    
    # Differential testing: CIN vs GS
    common_patients = exposure_df.index.intersection(meta_df.index)
    subtypes = meta_df.loc[common_patients, "MOLECULAR_SUBTYPE"]
    cin_mask = (subtypes == "CIN")
    gs_mask = (subtypes == "GS")
    
    diff_records = []
    for sig in sig_names:
        cin_vals = exposure_df.loc[cin_mask, sig]
        gs_vals = exposure_df.loc[gs_mask, sig]
        
        stat, pval = stats.ttest_ind(cin_vals, gs_vals, equal_var=False)
        diff_records.append({
            "Signature": sig,
            "mean_CIN": cin_vals.mean(),
            "mean_GS": gs_vals.mean(),
            "diff_CIN_vs_GS": cin_vals.mean() - gs_vals.mean(),
            "stat": stat,
            "pvalue": pval
        })
        
    diff_df = pd.DataFrame(diff_records).sort_values("pvalue", ascending=True)
    out_diff = os.path.join(RESULTS_DIR, "differential_mutational_signatures.csv")
    diff_df.to_csv(out_diff, index=False)
    print(f"Differential mutational signature testing complete. Saved to: {out_diff}")
    
    return exposure_df, diff_df, H

def plot_signature_figures(matrix_df, exposure_df, meta_df):
    """Generate 96-trinucleotide spectrum plot and signature exposure barplot."""
    # 1. 96-Trinucleotide Spectrum
    mean_spectrum = matrix_df.mean(axis=0)
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    
    colors = ["#1E90FF", "#000000", "#E41A1C", "#808080", "#4DAF4A", "#FF7F00"]
    channel_colors = []
    for i, sub in enumerate(SUBTYPES):
        channel_colors.extend([colors[i]] * 16)
        
    ax.bar(range(96), mean_spectrum, color=channel_colors, width=0.8)
    ax.set_xticks(range(0, 96, 4))
    ax.set_xticklabels([CHANNELS_96[i] for i in range(0, 96, 4)], rotation=90, fontsize=7)
    ax.set_title("Cohort Mean 96-Trinucleotide Somatic Mutational Spectrum", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Mean Mutation Count", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trinucleotide Context", fontsize=12, fontweight="bold")
    sns.despine()
    
    out_spec_fig = os.path.join(FIGURES_DIR, "mutational_spectrum_96.png")
    plt.tight_layout()
    plt.savefig(out_spec_fig, dpi=300)
    plt.close()
    print(f"Mutational spectrum figure saved to: {out_spec_fig}")
    
    # 2. Signature Exposure Barplot (CIN vs GS)
    plot_df = exposure_df.copy()
    plot_df["MOLECULAR_SUBTYPE"] = meta_df.loc[plot_df.index, "MOLECULAR_SUBTYPE"]
    melted = plot_df.melt(id_vars=["MOLECULAR_SUBTYPE"], var_name="Signature", value_name="Relative_Exposure")
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.barplot(
        data=melted,
        x="Signature",
        y="Relative_Exposure",
        hue="MOLECULAR_SUBTYPE",
        palette={"CIN": "#D95F02", "GS": "#1B9E77"},
        ax=ax
    )
    
    ax.set_title("COSMIC Mutational Signature Relative Exposures: CIN vs. GS", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Mutational Signature Process", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Relative Exposure Score", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontweight="bold")
    ax.legend(title="Molecular Subtype", frameon=True)
    sns.despine()
    
    out_exp_fig = os.path.join(FIGURES_DIR, "mutational_signature_exposures.png")
    plt.tight_layout()
    plt.savefig(out_exp_fig, dpi=300)
    plt.close()
    print(f"Mutational signature exposure figure saved to: {out_exp_fig}")

def main():
    meta_path = os.path.join(RESULTS_DIR, "gs_cin_sample_metadata.csv")
    mut_path = os.path.join(STAD_PUB_DIR, "data_mutations.txt")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}.")
        
    meta_df = pd.read_csv(meta_path, index_col=0)
    
    print("Step 1: Constructing 96-trinucleotide mutational spectrum matrix...")
    matrix_df = build_96_matrix(mut_path, meta_df)
    
    print("\nStep 2: Deconvolving COSMIC mutational signatures via NMF...")
    exposure_df, diff_df, H = deconvolve_cosmic_signatures(matrix_df, meta_df)
    
    print("\nStep 3: Generating mutational signature figures...")
    plot_signature_figures(matrix_df, exposure_df, meta_df)
    
    print("\nCOSMIC Mutational Signatures pipeline executed successfully!")

if __name__ == "__main__":
    main()
