# Comparative Multi-Omics & Clinical Profiling: Genomically Stable (GS) vs. Chromosomal Instable (CIN) Gastric Cancer

[![Project Status: In Progress](https://img.shields.io/badge/Status-Phase%201%20Planning-blue.svg)](#project-phases)
[![Dataset](https://img.shields.io/badge/Dataset-TCGA--STAD%20(Nature%202014)-green.svg)](https://www.nature.com/articles/nature13480)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🔬 Overview & Scientific Motivation

Gastric cancer (Stomach Adenocarcinoma, **STAD**) is a heterogeneous disease with distinct molecular subtypes characterized by landmark studies, notably The Cancer Genome Atlas (**TCGA**, Nature 2014). The TCGA classification defines four primary molecular subtypes:
1. **EBV** (Epstein–Barr virus-positive)
2. **MSI** (Microsatellite Instability)
3. **GS** (Genomically Stable)
4. **CIN** (Chromosomal Instability)

While **EBV** and **MSI** subtypes display distinct hypermutation phenotypes, high immunogenicity, and specific clinical trajectories, the majority of non-hypermutated gastric adenocarcinomas fall into either the **Genomically Stable (GS)** or **Chromosomal Instability (CIN)** classes:
- **Genomically Stable (GS)**: Enriched for diffuse histological variants, frequent *CDH1* and *RHOA* mutations, *CLDN18-ARHGAP* fusions, and low somatic copy-number alterations.
- **Chromosomal Instability (CIN)**: Characterized by marked aneuploidy, recurrent focal amplifications in receptor tyrosine kinases (e.g., *HER2/ERBB2*, *EGFR*, *MET*, *FGFR2*) and cell cycle regulators (e.g., *CCNE1*, *CCND1*, *CDK6*), and frequent *TP53* mutations.

This project focuses on a deep, comparative multi-omics and clinical investigation comparing **GS vs. CIN** subtypes, uncovering mechanistic drivers, survival disparities, epigenetic landscapes, transcription factor regulatory networks, and training robust machine learning models to classify and distinguish these subtypes from multi-modal molecular features.

---

## 🗺️ Project Roadmap & Phased Implementation

The project is structured into three structured, sequential phases:

```mermaid
flowchart TD
    subgraph Phase1["Phase 1: Clinical Cohort & Survival Analysis"]
        A[Raw TCGA STAD Cohort] --> B[Filter & Exclude EBV & MSI Subtypes]
        B --> C[Stratify Cohort: GS vs CIN]
        C --> D[Clinical Covariates & Demographic Profiling]
        C --> E[Kaplan-Meier Survival Analysis]
        E --> F[Log-Rank Test p-value & Hazard / Log Ratio Calculation]
    end

    subgraph Phase2["Phase 2: Multi-Omics Feature Extraction & TF Profiling"]
        G[DNA Methylation] --> H[Differentially Methylated Regions DMRs]
        I[mRNA Expression] --> J[Differential Gene Expression DGE]
        I --> K[decoupleR + DoRothEA: TF Activity Inference]
        L[Copy Number Alteration] --> M[GISTIC 2.0 Peak / Segment Analysis]
        N[Somatic Mutations] --> O[COSMIC Mutational Signatures Deconvolution]
        H & J & K & M & O --> P[Integrated Multi-Omics Feature Matrix]
    end

    subgraph Phase3["Phase 3: Predictive Modeling, Validation & Testing"]
        P --> Q[Multi-modal Feature Preprocessing & Selection]
        Q --> R[Model Training: ElasticNet, Random Forest, XGBoost, Neural Nets]
        R --> S[Stratified K-Fold & Repeated Cross-Validation]
        S --> T[Evaluation: ROC-AUC, PR-AUC, F1-Score, Brier Score]
        S --> U[Model Explainability: SHAP Analysis & Biomarker Ranking]
    end

    Phase1 --> Phase2 --> Phase3
```

---

### 📌 Phase 1: Clinical Data Preprocessing & Survival Profiling
- **Cohort Curation**:
  - Ingest TCGA STAD clinical patient and sample metadata (`data_clinical_patient.txt`, `data_clinical_sample.txt`).
  - Filter out and drop samples belonging to **EBV** and **MSI** molecular subtypes.
  - Cohort validation: Retain strictly curated **GS** and **CIN** cohorts with high-confidence subtype annotations.
- **Baseline Clinical Comparison**:
  - Evaluate clinicopathological parameters (age, sex, AJCC tumor stage, TNM staging, Lauren classification: diffuse vs. intestinal, anatomic tumor location).
- **Survival Analysis**:
  - Kaplan–Meier survival curve estimation for Overall Survival (OS) and Disease-Free Survival (DFS).
  - Statistical hypothesis testing via **Log-Rank test** (deriving exact $p$-values).
  - Compute Hazard Ratios (HR) and Log-Hazard Ratios using univariate and multivariable **Cox Proportional Hazards Regression** adjusting for potential confounding covariates (e.g., stage, age).

---

### 📌 Phase 2: Multi-Omics Feature Extraction & Regulatory Signatures
- **Epigenomics (DNA Methylation)**:
  - Identify Differentially Methylated Regions (**DMRs**) and CpG probes distinguishing GS from CIN.
  - Characterize hyper/hypo-methylation patterns across promoter and enhancer regions.
- **Transcriptomics (Differential Gene Expression - DGE)**:
  - Perform DGE analysis (Log2 Fold Change, Adjusted $p$-values / FDR thresholds).
  - Pathway and Gene Ontology (GO) / GSEA enrichment analysis.
- **Gene Regulatory Network (GRN) & Transcription Factor Activity**:
  - Leverage **`decoupleR`** coupled with the **`DoRothEA`** regulon database.
  - Infer transcription factor (TF) regulatory activity scores per sample to capture functional upstream transcriptional switches differentiating GS and CIN.
- **Copy Number Alterations (CNA)**:
  - High-resolution copy-number segmentation and **GISTIC 2.0** focal/arm-level amplification & deletion profiling.
  - Quantify CIN burden (fraction of genome altered) vs. diploid-like genomic stability in GS.
- **Mutational Signatures (COSMIC)**:
  - Deconvolve somatic single nucleotide variants (SNVs) against **COSMIC Mutational Signatures** (v2/v3).
  - Evaluate signature activities (e.g., Aging/Deamination, HRD, APOBEC, etc.).
- **Multi-Omics Fusion**:
  - Standardize and concatenate multi-omics feature matrices into a clean, aligned tabular feature store.

---

### 📌 Phase 3: Model Architecture, Training, Benchmarking & Testing
- **Predictive Modeling**:
  - Build machine learning classifiers to accurately separate GS from CIN tumors using single-omics and integrated multi-omics features.
  - Algorithms: Regularized Logistic Regression / ElasticNet, Random Forest, XGBoost / LightGBM, and Support Vector Machines (SVM).
- **Validation Strategy**:
  - Stratified $K$-Fold cross-validation and repeated nested CV to prevent data leakage and overfitting.
- **Performance Benchmarks**:
  - Evaluate via Receiver Operating Characteristic (ROC-AUC), Precision-Recall Curve (PR-AUC), Balanced Accuracy, Sensitivity, Specificity, and F1-score.
- **Interpretability & Biomarker Prioritization**:
  - Global and local feature attribution using **SHAP (SHapley Additive exPlanations)** values.
  - Identify key multi-omics driver genes, regulatory TFs, and genomic hallmarks that define the distinct biology of GS vs. CIN gastric cancer.

---

## 📂 Repository Structure

```text
GS_vs_CIN/
├── README.md                           # Project documentation and roadmap
├── .gitignore                          # Git ignore rules for data and artifacts
├── data/                               # Data directory (local TCGA datasets, ignored from git)
│   └── stad_tcga_pub/                  # TCGA STAD Nature 2014 dataset
├── notebooks/                          # Interactive Jupyter analysis notebooks
│   ├── 01_phase1_clinical_survival.ipynb
│   ├── 02_phase2_multiomics_features.ipynb
│   └── 03_phase3_modeling_evaluation.ipynb
├── src/                                # Modular source code (Python / R)
│   ├── __init__.py
│   ├── clinical/                       # Clinical curation & survival modeling
│   ├── omics/                          # DGE, methylation, GISTIC & COSMIC signatures
│   ├── tf_inference/                   # DecoupleR & DoRothEA TF activity calculations
│   ├── models/                         # ML architectures, cross-validation & evaluation
│   └── utils/                          # Common helpers, I/O and plotting utilities
├── results/                            # Generated survival curves, volcano plots, metrics
└── figures/                            # Publication-quality figures & diagrams
```

---

## ⚙️ Tech Stack & Dependencies

- **Language & Environment**: Python 3.10+, R 4.x (optional for specialized Bioconductor packages)
- **Survival Analysis**: `lifelines`, `scikit-survival`, `statsmodels`
- **Transcriptomics & TF Inference**: `decoupleR`, `pydeseq2`, `scanpy` / `anndata`, `biomart`
- **Machine Learning & Evaluation**: `scikit-learn`, `xgboost`, `lightgbm`, `optuna`
- **Explainability**: `shap`, `lime`
- **Data Wrangling & Visualization**: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `plotly`

---

## 📜 References & Acknowledgements

1. **TCGA STAD Publication**: The Cancer Genome Atlas Research Network. *Comprehensive molecular characterization of gastric adenocarcinoma.* Nature **513**, 202–209 (2014). [doi:10.1038/nature13480](https://doi.org/10.1038/nature13480).
2. **DoRothEA Regulons**: Garcia-Alonso, L. et al. *Benchmark and integration of resources for the estimation of human transcription factor activities.* Genome Research **29**, 1363–1375 (2019).
3. **DecoupleR**: Badia-i-Mompel, P. et al. *decoupleR: ensemble of methods to infer biological activities from omics data.* Bioinformatics Advances **2**, vbac016 (2022).
4. **COSMIC Mutational Signatures**: Alexandrov, L. B. et al. *Signatures of mutational processes in human cancer.* Nature **500**, 415–421 (2013).
