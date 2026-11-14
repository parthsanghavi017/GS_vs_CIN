# Phase 4: mRNA + Transcription Factors + Mutation Signatures
## GS vs. CIN Classification — TCGA-STAD

---

## What Did We Do In Simple Words?

We tried to answer one question:

> **"Can we tell apart CIN tumors from GS tumors using ONLY gene expression + TF activity + mutation patterns — without looking at any chromosome copy number data?"**

We used **3 types of signals**, combined into 81 features total, to train machine learning models:

| Signal Type | What It Measures | Features Used |
|:---|:---|:---:|
| mRNA Gene Expression | Which genes are turned ON/OFF in each tumor | 46 genes |
| TF Regulon Activity | Which master "switch" proteins (transcription factors) are active | 30 TFs |
| COSMIC Mutational Signatures | What kind of DNA damage happened in the tumor over its lifetime | 5 SBS signatures |
| **TOTAL** | | **81 features** |

No CNA (copy number) data was used. No CNV data was used.

---

## What Are Mutational Signatures? (Simple Explanation)

Think of your DNA like a book. Over your lifetime, the pages get damaged — words get misspelled or erased. Different things cause different kinds of misspellings:

- Too much sun causes a specific kind of spelling mistake (C>T at TpC sites)
- APOBEC enzymes (which go haywire in chromosomally unstable tumors) cause yet another pattern

Scientists have catalogued these patterns. They are called COSMIC Signatures (Single Base Substitution / SBS). By reading the pattern of mutations in a tumor, we can figure out which biological forces damaged that tumor DNA.

We used 5 signatures most relevant to gastric cancer:

| Signature | What Caused It | Higher In |
|:---|:---|:---:|
| SBS1 — Aging Clock | Normal DNA aging over decades | GS |
| SBS3 — DNA Repair | Broken DNA repair machinery (like BRCA) | GS slightly |
| SBS5 — Clock-like | Slow background replication errors | Both |
| SBS17 — Gastric/5-FU | Gastric cancer environment + chemotherapy | CIN |
| SBS2_13 — APOBEC | APOBEC enzymes during chromosomal instability | CIN (very strongly!) |

The key one is APOBEC (SBS2_13): CIN tumors had twice as much APOBEC mutation as GS tumors (mean 0.221 vs 0.104, p = 1.37e-07).

---

## Results: How Well Did It Work?

Test Set: 36 patients (25 CIN, 11 GS) — never seen by the model during training

### Model Performance Table

| Model | 10-Fold CV AUC | Test ROC-AUC | Test PR-AUC | Balanced Accuracy | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random Forest | 0.891 | **0.975** | **0.990** | 0.798 | **0.906** |
| SVM | 0.890 | **0.945** | 0.976 | **0.844** | **0.923** |
| LightGBM | 0.876 | **0.949** | 0.980 | 0.758 | 0.863 |
| XGBoost | 0.865 | 0.920 | 0.964 | 0.804 | 0.880 |
| Logistic Regression | 0.847 | 0.913 | 0.963 | 0.778 | 0.885 |

Best single classifier: Random Forest — Test ROC-AUC = 0.975

---

## Confusion Matrix Breakdown (Who Did It Get Wrong?)

On 36 holdout test patients:

| Model | CIN Found (of 25) | CIN Missed | GS Found (of 11) | GS Called CIN | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random Forest | 24 (96%) | 1 | 7 (64%) | 4 | 86.1% |
| SVM | 24 (96%) | 1 | 8 (73%) | 3 | 88.9% |
| XGBoost | 22 (88%) | 3 | 8 (73%) | 3 | 83.3% |
| LightGBM | 22 (88%) | 3 | 7 (64%) | 4 | 80.6% |
| Logistic Regression | 23 (92%) | 2 | 7 (64%) | 4 | 83.3% |

Key takeaway: The model is very good at finding CIN tumors (misses only 1 of 25).
It is less perfect with GS — sometimes a GS tumor looks "CIN-like" in RNA because of secondary proliferation stress.

---

## Did Mutation Signatures Help vs. Just mRNA + TF?

| Regime | Features | Best Test ROC-AUC | Best F1 |
|:---|:---:|:---:|:---:|
| Pure mRNA only | 46 | 0.909 | 0.852 |
| mRNA + TF only | 76 | 0.941 | 0.906 |
| mRNA + TF + SBS (this analysis) | **81** | **0.975** | **0.923** |

YES — adding mutation signatures boosted the test ROC-AUC from 0.941 to 0.975!

---

## SBS Signature Statistics (All 177 Patients)

| Signature | CIN Mean | GS Mean | p-value | Significant? |
|:---|:---:|:---:|:---:|:---:|
| SBS2_13 APOBEC | 0.221 | 0.104 | 1.37e-07 | YES (very strong) |
| SBS17 Gastric | 0.142 | 0.085 | 9.16e-03 | YES |
| SBS3 HRD | 0.523 | 0.594 | 1.51e-02 | YES (GS is higher) |
| SBS1 Aging | 0.029 | 0.043 | 0.518 | No |
| SBS5 Clock | 0.077 | 0.097 | 0.876 | No |

---

## Files Generated

### results/
- ml_mrna_tf_sbs_metrics.csv — Full model metrics (CV + test)
- confusion_matrix_detail.csv — Per-class TP/FP/FN/TN per model
- sbs_differential_statistics.csv — Mann-Whitney stats per SBS signature

### data/
- mrna_tf_sbs_feature_matrix.csv — Complete 81-feature patient matrix (177 patients)

### figures/
- phase4_roc_pr.png — ROC + PR curves for all 5 models
- phase4_confusion_matrices.png — Confusion matrices on holdout test
- phase4_sbs_distributions.png — Boxplots of SBS exposures in GS vs CIN
- phase4_regime_comparison.png — Bar chart: mRNA vs mRNA+TF vs mRNA+TF+SBS

---

## One Paragraph Summary

We built a machine learning classifier using 81 biological signals (46 mRNA + 30 TFs + 5 mutation signatures) to classify TCGA-STAD patients as GS or CIN without any copy number data. On a holdout test set of 36 patients, Random Forest achieved Test ROC-AUC = 0.975 and F1 = 0.906, finding 24 of 25 CIN patients correctly. Adding mutation signatures (especially APOBEC — SBS2_13, which is significantly elevated in CIN at p = 1.37e-07) on top of mRNA + TF boosted test ROC-AUC from 0.941 to 0.975, showing that the APOBEC mutational footprint provides a powerful complementary signal orthogonal to gene expression.

---
Analysis script: run_phase4_rna_tf_mutsig.py
