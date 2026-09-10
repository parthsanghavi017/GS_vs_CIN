import os
import io
import gzip
import json
import requests
import pandas as pd
import numpy as np

BASE_DIR = "/media/parth/Research_Volume/GS_vs_CIN"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STAD_PUB_DIR = os.path.join(BASE_DIR, "stad_tcga_pub")

def load_gs_cin_cohort():
    """Load curated GS and CIN samples from TCGA STAD clinical metadata."""
    patient_path = os.path.join(STAD_PUB_DIR, "data_clinical_patient.txt")
    sample_path = os.path.join(STAD_PUB_DIR, "data_clinical_sample.txt")
    
    patient_df = pd.read_csv(patient_path, sep="\t", skiprows=4)
    sample_df = pd.read_csv(sample_path, sep="\t", skiprows=4)
    
    merged = sample_df.merge(patient_df, on="PATIENT_ID", how="inner")
    cohort = merged[merged["MOLECULAR_SUBTYPE"].isin(["GS", "CIN"])].copy()
    cohort["PATIENT_BARCODE_SHORT"] = cohort["PATIENT_ID"].str[:12]
    return cohort

def fetch_gdc_star_count_files(cohort_df):
    """
    Query GDC API for TCGA-STAD STAR - Counts files and match with GS/CIN patient barcodes.
    """
    gdc_files_url = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-STAD"}},
            {"op": "=", "content": {"field": "files.analysis.workflow_type", "value": "STAR - Counts"}},
            {"op": "=", "content": {"field": "files.data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "files.access", "value": "open"}}
        ]
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type",
        "format": "JSON",
        "size": "1000"
    }
    
    response = requests.get(gdc_files_url, params=params)
    response.raise_for_status()
    hits = response.json()["data"]["hits"]
    
    print(f"Retrieved {len(hits)} STAR count file records from GDC API.")
    
    # Map patient barcode (12 chars, e.g. TCGA-BR-4255) to GDC file info
    patient_barcodes = set(cohort_df["PATIENT_BARCODE_SHORT"])
    matched_hits = []
    
    for hit in hits:
        cases = hit.get("cases", [])
        if not cases:
            continue
        case = cases[0]
        patient_barcode = case.get("submitter_id", "")[:12]
        sample = case.get("samples", [{}])[0]
        sample_type = sample.get("sample_type", "")
        sample_barcode = sample.get("submitter_id", "")
        
        # Prefer Primary Tumor samples
        if patient_barcode in patient_barcodes and "Tumor" in sample_type:
            hit_info = {
                "file_id": hit["file_id"],
                "file_name": hit["file_name"],
                "patient_barcode": patient_barcode,
                "sample_barcode": sample_barcode,
                "sample_type": sample_type
            }
            matched_hits.append(hit_info)
            
    matched_df = pd.DataFrame(matched_hits)
    # Drop duplicates if multiple tumor aliquots exist, keeping first
    matched_df = matched_df.drop_duplicates(subset=["patient_barcode"]).reset_index(drop=True)
    
    print(f"Matched {len(matched_df)} unique patient STAR count files to GS/CIN cohort.")
    return matched_df

from concurrent.futures import ThreadPoolExecutor, as_completed

def _download_single_file(row, cache_dir):
    file_id = row["file_id"]
    patient_barcode = row["patient_barcode"]
    local_path = os.path.join(cache_dir, f"{file_id}_{row['file_name']}")
    
    # Download if missing or empty
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 1000:
        data_url = f"https://api.gdc.cancer.gov/data/{file_id}"
        r = requests.get(data_url, stream=True, timeout=30)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
            
    if local_path.endswith(".gz"):
        with gzip.open(local_path, "rt") as f:
            lines = f.readlines()
    else:
        with open(local_path, "r") as f:
            lines = f.readlines()
            
    valid_lines = [l for l in lines if not l.startswith("#") and not l.startswith("N_")]
    if not valid_lines:
        # Re-download if corrupt
        data_url = f"https://api.gdc.cancer.gov/data/{file_id}"
        r = requests.get(data_url, stream=True, timeout=30)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        if local_path.endswith(".gz"):
            with gzip.open(local_path, "rt") as f:
                lines = f.readlines()
        else:
            with open(local_path, "r") as f:
                lines = f.readlines()
        valid_lines = [l for l in lines if not l.startswith("#") and not l.startswith("N_")]

    df_file = pd.read_csv(io.StringIO("".join(valid_lines)), sep="\t")
    gene_info = df_file[["gene_id", "gene_name", "gene_type"]].copy()
    counts = df_file["unstranded"].values
    
    return patient_barcode, gene_info, counts

def download_and_parse_counts(matched_df, cache_dir=None):
    """
    Download TSV count files from GDC API in parallel and extract unstranded counts.
    """
    if cache_dir is None:
        cache_dir = os.path.join(DATA_DIR, "gdc_star_counts")
    os.makedirs(cache_dir, exist_ok=True)
    
    count_dict = {}
    gene_info = None
    
    print(f"Downloading & parsing {len(matched_df)} count files in parallel (20 threads)...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_download_single_file, row, cache_dir): idx for idx, row in matched_df.iterrows()}
        completed = 0
        for future in as_completed(futures):
            patient_barcode, g_info, counts = future.result()
            if gene_info is None:
                gene_info = g_info
            count_dict[patient_barcode] = counts
            completed += 1
            if completed % 25 == 0 or completed == len(matched_df):
                print(f"Downloaded & parsed {completed}/{len(matched_df)} files...")
                
    counts_df = pd.DataFrame(count_dict)
    counts_df.index = gene_info["gene_name"]
    counts_df = counts_df.loc[counts_df.index.dropna()]
    counts_df = counts_df.groupby(counts_df.index).max()
    
    return counts_df, gene_info

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Step 1: Loading GS vs CIN clinical cohort metadata...")
    cohort = load_gs_cin_cohort()
    print(f"Cohort loaded: {len(cohort)} total samples.")
    
    print("\nStep 2: Fetching STAR count file manifest from GDC API...")
    matched_df = fetch_gdc_star_count_files(cohort)
    
    print("\nStep 3: Downloading & parsing unstranded raw counts...")
    counts_df, gene_info = download_and_parse_counts(matched_df)
    
    # Align cohort annotations with downloaded count matrix columns
    cohort_aligned = cohort.merge(matched_df, left_on="PATIENT_BARCODE_SHORT", right_on="patient_barcode", how="inner")
    cohort_aligned = cohort_aligned.drop_duplicates(subset=["patient_barcode"]).set_index("patient_barcode")
    
    # Align matrix columns
    common_patients = [p for p in cohort_aligned.index if p in counts_df.columns]
    counts_df = counts_df[common_patients]
    cohort_aligned = cohort_aligned.loc[common_patients]
    
    out_counts_path = os.path.join(DATA_DIR, "tcga_stad_unstranded_counts.csv")
    out_meta_path = os.path.join(RESULTS_DIR, "gs_cin_sample_metadata.csv")
    
    counts_df.to_csv(out_counts_path)
    cohort_aligned.to_csv(out_meta_path)
    
    print(f"\nSaved unstranded count matrix: {counts_df.shape} (genes x samples) to {out_counts_path}")
    print(f"Saved aligned metadata: {cohort_aligned.shape} samples to {out_meta_path}")
    print("Subtype distribution in dataset:")
    print(cohort_aligned["MOLECULAR_SUBTYPE"].value_counts())

if __name__ == "__main__":
    main()
