import os
import nbformat as nbf

BASE_DIR = "/media/parth/Research_Volume/GS_vs_CIN"
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title cell
    cells.append(nbf.v4.new_markdown_cell("""# Phase 2: Transcriptomic DGE, GSEA, decoupleR+DoRothEA TF Activity, CNV, Mutational Signatures & SV Profiling

**Project:** Comparative Multi-Omics & Clinical Profiling: Genomically Stable (GS) vs. Chromosomal Instable (CIN) Gastric Cancer  
**Dataset:** TCGA-STAD (*Nature* 2014) & GDC STAR Unstranded RNA-seq Counts ($N=190$ samples: $136$ CIN, $54$ GS)  

---

## 🔬 Overview & Key Objectives
This notebook executes **Phase 2** of the multi-omics workflow:
1. **Unstranded STAR Count Ingestion**: Load raw integer count matrix directly from GDC API / TCGABiolinks aligned with curated GS vs. CIN clinical cohort.
2. **PyDESeq2 Differential Expression (DGE)**: Fit negative binomial generalized linear models (GLMs) to derive $\\log_2(\\text{Fold Change})$, Wald statistics, and Benjamini-Hochberg adjusted $p$-values ($padj/FDR$).
3. **Gene Set Enrichment Analysis (GSEA)**: Perform pre-ranked GSEA against MSigDB Hallmark gene sets to identify dysregulated cellular pathways.
4. **Transcription Factor (TF) Regulatory Activity Scores**: Leverage `decoupleR` coupled with human `DoRothEA` regulons (confidence levels A, B, C) to infer sample-wise TF activity scores and isolate master regulatory switches.
5. **Copy Number Alterations (CNV)**: Compute Fraction Genome Altered (FGA) burden and profile high-level amplifications ($+2$) and deletions ($-2$) across key driver genes (*ERBB2*, *CCNE1*, *CDK6*, *EGFR*, *MET*, *FGFR2*, *MYC*, *KRAS*, *TP53*, *CDKN2A*).
6. **COSMIC Mutational Signatures (SBS)**: Extract 96-trinucleotide substitution contexts and deconvolve COSMIC SBS signature exposures using NMF.
7. **Structural Variants & Gene Fusions (SV)**: Profile hallmark gastric cancer fusion drivers (`CLDN18` fusions, `MYLK` fusions, `CDKN2A` silencing).
8. **Integrated Multi-Modal Feature Store**: Construct a clean, concatenated multi-omics feature matrix (`results/phase2_integrated_feature_store.csv`) for Phase 3 machine learning classifiers.
"""))

    # Cell 1: Environment & Imports
    cells.append(nbf.v4.new_markdown_cell("### 1. Environment & Library Dependencies"))
    cells.append(nbf.v4.new_code_cell("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

BASE_DIR = "/media/parth/Research_Volume/GS_vs_CIN"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

print("Environment loaded successfully.")
"""))

    # Cell 2: Data Ingestion
    cells.append(nbf.v4.new_markdown_cell("### 2. Ingest TCGA-STAD Unstranded STAR Counts & Clinical Cohort Metadata"))
    cells.append(nbf.v4.new_code_cell("""counts_path = os.path.join(DATA_DIR, "tcga_stad_unstranded_counts.csv")
meta_path = os.path.join(RESULTS_DIR, "gs_cin_sample_metadata.csv")

counts_df = pd.read_csv(counts_path, index_col=0)
meta_df = pd.read_csv(meta_path, index_col=0)

print(f"Unstranded Count Matrix shape: {counts_df.shape[0]} genes x {counts_df.shape[1]} samples.")
print(f"Sample Metadata shape: {meta_df.shape[0]} samples.")
print("\\nCohort Subtype Distribution:")
print(meta_df["MOLECULAR_SUBTYPE"].value_counts().to_string())
"""))

    # Cell 3: DESeq2 DGE Results
    cells.append(nbf.v4.new_markdown_cell("### 3. Differential Gene Expression Analysis (DESeq2)"))
    cells.append(nbf.v4.new_code_cell("""deg_path = os.path.join(RESULTS_DIR, "dge_deseq2_gs_vs_cin.csv")
deg_df = pd.read_csv(deg_path, index_col=0)

print("Top 10 Upregulated Genes in CIN (Chromosomal Instability):")
display(deg_df[deg_df["log2FoldChange"] > 0].sort_values("padj").head(10)[["baseMean", "log2FoldChange", "stat", "pvalue", "padj"]])

print("\\nTop 10 Upregulated Genes in GS (Genomically Stable):")
display(deg_df[deg_df["log2FoldChange"] < 0].sort_values("padj").head(10)[["baseMean", "log2FoldChange", "stat", "pvalue", "padj"]])
"""))

    # Cell 4: Display Volcano Plot
    cells.append(nbf.v4.new_markdown_cell("#### Volcano Plot Visualization (CIN vs. GS)"))
    cells.append(nbf.v4.new_code_cell("""volcano_fig = os.path.join(FIGURES_DIR, "volcano_dge_deseq2.png")
if os.path.exists(volcano_fig):
    display(Image(filename=volcano_fig))
"""))

    # Cell 5: GSEA Results
    cells.append(nbf.v4.new_markdown_cell("### 4. Gene Set Enrichment Analysis (GSEA - MSigDB Hallmark)"))
    cells.append(nbf.v4.new_code_cell("""gsea_path = os.path.join(RESULTS_DIR, "gsea_hallmark_results.csv")
gsea_df = pd.read_csv(gsea_path)

print("Top Enriched MSigDB Hallmark Pathways:")
display(gsea_df.head(15)[["Term", "NES", "NOM p-val", "FDR q-val"]])
"""))

    # Cell 6: Display GSEA Plot
    cells.append(nbf.v4.new_markdown_cell("#### GSEA Hallmark Pathways Barplot"))
    cells.append(nbf.v4.new_code_cell("""gsea_fig = os.path.join(FIGURES_DIR, "gsea_hallmark_enr.png")
if os.path.exists(gsea_fig):
    display(Image(filename=gsea_fig))
"""))

    # Cell 7: decoupleR + DoRothEA TF Activity Scoring
    cells.append(nbf.v4.new_markdown_cell("### 5. Transcription Factor Regulatory Activity Scoring (decoupleR + DoRothEA)"))
    cells.append(nbf.v4.new_code_cell("""tf_acts_path = os.path.join(RESULTS_DIR, "tf_activity_scores.csv")
diff_tf_path = os.path.join(RESULTS_DIR, "differential_tf_activity_gs_vs_cin.csv")

tf_acts = pd.read_csv(tf_acts_path, index_col=0)
diff_tf = pd.read_csv(diff_tf_path)

print(f"Sample-wise TF Activity Matrix shape: {tf_acts.shape} (samples x TFs).")
print("\\nTop 10 Most Active TFs in CIN Tumors:")
display(diff_tf[diff_tf["diff_CIN_vs_GS"] > 0].head(10)[["TF", "mean_CIN", "mean_GS", "diff_CIN_vs_GS", "pvalue", "padj"]])

print("\\nTop 10 Most Active TFs in GS Tumors:")
display(diff_tf[diff_tf["diff_CIN_vs_GS"] < 0].head(10)[["TF", "mean_CIN", "mean_GS", "diff_CIN_vs_GS", "pvalue", "padj"]])
"""))

    # Cell 8: Display TF Heatmap & Volcano Plot
    cells.append(nbf.v4.new_markdown_cell("#### TF Activity Visualizations"))
    cells.append(nbf.v4.new_code_cell("""tf_heatmap = os.path.join(FIGURES_DIR, "tf_activity_heatmap.png")
tf_volcano = os.path.join(FIGURES_DIR, "tf_activity_volcano.png")

if os.path.exists(tf_heatmap):
    print("TF Regulatory Activity Clustered Heatmap:")
    display(Image(filename=tf_heatmap))

if os.path.exists(tf_volcano):
    print("Differential TF Regulatory Activity Volcano Plot:")
    display(Image(filename=tf_volcano))
"""))

    # Cell 9: Copy Number Alterations (CNV) Profiling
    cells.append(nbf.v4.new_markdown_cell("### 6. Copy Number Alterations (CNV) & Fraction Genome Altered (FGA)"))
    cells.append(nbf.v4.new_code_cell("""fga_path = os.path.join(RESULTS_DIR, "cnv_fga_burden.csv")
focal_path = os.path.join(RESULTS_DIR, "cnv_focal_alterations.csv")

fga_df = pd.read_csv(fga_path, index_col=0)
focal_df = pd.read_csv(focal_path)

print(f"FGA Burden Dataset shape: {fga_df.shape} (samples x metrics).")
print("\\nMean FGA by Subtype:")
print(fga_df.groupby("MOLECULAR_SUBTYPE")["FGA"].mean().to_string())

print("\\nTop Focal Driver Alteration Frequencies (% High Amplification +2):")
display(focal_df.head(10))

fga_fig = os.path.join(FIGURES_DIR, "cnv_fga_distribution.png")
driver_fig = os.path.join(FIGURES_DIR, "cnv_alteration_frequencies.png")

if os.path.exists(fga_fig):
    display(Image(filename=fga_fig))
if os.path.exists(driver_fig):
    display(Image(filename=driver_fig))
"""))

    # Cell 10: Mutational Signatures (COSMIC SBS)
    cells.append(nbf.v4.new_markdown_cell("### 7. Somatic Mutational Signatures (COSMIC SBS 96-Trinucleotide Deconvolution)"))
    cells.append(nbf.v4.new_code_cell("""mut_sig_path = os.path.join(RESULTS_DIR, "mutational_signature_activities.csv")
diff_sig_path = os.path.join(RESULTS_DIR, "differential_mutational_signatures.csv")

mut_sig_df = pd.read_csv(mut_sig_path, index_col=0)
diff_sig_df = pd.read_csv(diff_sig_path)

print(f"Mutational Signature Exposures shape: {mut_sig_df.shape} (samples x signatures).")
print("\\nDifferential Mutational Signature Exposures (CIN vs GS):")
display(diff_sig_df)

spec_fig = os.path.join(FIGURES_DIR, "mutational_spectrum_96.png")
exp_fig = os.path.join(FIGURES_DIR, "mutational_signature_exposures.png")

if os.path.exists(spec_fig):
    display(Image(filename=spec_fig))
if os.path.exists(exp_fig):
    display(Image(filename=exp_fig))
"""))

    # Cell 11: Structural Variants & Gene Fusions
    cells.append(nbf.v4.new_markdown_cell("### 8. Structural Variants & Gene Fusions (data_sv.txt)"))
    cells.append(nbf.v4.new_code_cell("""sv_matrix_path = os.path.join(RESULTS_DIR, "sv_events_matrix.csv")
sv_summary_path = os.path.join(RESULTS_DIR, "structural_variant_summary.csv")

sv_matrix_df = pd.read_csv(sv_matrix_path, index_col=0)
sv_summary_df = pd.read_csv(sv_summary_path)

print(f"Structural Variant Matrix shape: {sv_matrix_df.shape} (samples x events).")
print("\\nStructural Variant & Gene Fusion Summary:")
display(sv_summary_df)

sv_fig = os.path.join(FIGURES_DIR, "structural_variants_distribution.png")
if os.path.exists(sv_fig):
    display(Image(filename=sv_fig))
"""))

    # Cell 12: Build Integrated Multi-Modal Feature Store
    cells.append(nbf.v4.new_markdown_cell("### 9. Export Integrated Multi-Modal Feature Matrix Store for Phase 3"))
    cells.append(nbf.v4.new_code_cell("""# Combine TF Activity Scores, FGA Burden, Mutational Signature Exposures, and SV Events
common_samples = tf_acts.index.intersection(meta_df.index)

feature_store = tf_acts.loc[common_samples].copy()

# Add FGA
if "FGA" in fga_df.columns:
    feature_store["FGA_Burden"] = fga_df["FGA"].reindex(common_samples).fillna(0.0)

# Add Mutational Signature Exposures
for col in mut_sig_df.columns:
    feature_store[f"MutSig_{col}"] = mut_sig_df[col].reindex(common_samples).fillna(0.0)

# Add Structural Variants & Fusions
for col in sv_matrix_df.columns:
    feature_store[col] = sv_matrix_df[col].reindex(common_samples).fillna(0)

# Add Target Labels
feature_store["MOLECULAR_SUBTYPE"] = meta_df.loc[common_samples, "MOLECULAR_SUBTYPE"]
feature_store["SUBTYPE_CIN_VS_GS"] = (feature_store["MOLECULAR_SUBTYPE"] == "CIN").astype(int)

out_store_path = os.path.join(RESULTS_DIR, "phase2_integrated_feature_store.csv")
feature_store.to_csv(out_store_path)

print(f"Integrated Multi-Modal Phase 2 Feature Store exported successfully to: {out_store_path}")
print(f"Final shape: {feature_store.shape[0]} samples x {feature_store.shape[1]} features.")
"""))

    nb['cells'] = cells
    out_nb_path = os.path.join(NOTEBOOKS_DIR, "02_phase2_multiomics_features.ipynb")
    with open(out_nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated expanded notebook: {out_nb_path}")

if __name__ == "__main__":
    build_notebook()
