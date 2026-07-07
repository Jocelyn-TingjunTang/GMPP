"""
GMPP Data Consolidation Script (Auto-Detect Version)
=======================================================
Automatically scans a folder for GMPP CSV files, detects the year from
each filename, detects whether the file is in "wide" (transposed) or
"long" (standard) format, and consolidates everything into one clean
dataset ready for Power BI and machine learning.

No need to maintain a filename-to-year mapping by hand — just put all
your CSV files in the same folder as this script (or set DATA_FOLDER
below) and run it. Renaming files is safe as long as a 4-digit year
(20xx) appears somewhere in the filename.

Usage:
    python gmpp_consolidate.py

Output:
    GMPP_Consolidated.csv  — full combined dataset
    GMPP_Construction.csv  — Infrastructure & Construction projects only (for ML)
"""

import pandas as pd
import numpy as np
import os
import re
import glob
import warnings
warnings.filterwarnings('ignore')

# ── SETTINGS ──────────────────────────────────────────────────────────────────

# Folder containing all the GMPP CSV files. "." means "same folder as this script".
DATA_FOLDER = "."

# OPTIONAL: manually override the year for specific files.
# If a filename appears here, this year is used INSTEAD of auto-detecting
# it from the filename. Useful if a filename's year is ambiguous, wrong,
# or you simply don't want to rename files.
#
# Example:
# MANUAL_YEAR_OVERRIDE = {
#     "Annual_Report_-_consolidated_data_and_narratives.csv": 2016,
#     "some_confusing_filename.csv": 2020,
# }
MANUAL_YEAR_OVERRIDE = {
    # "filename.csv": year,
}

# ── COLUMN MAPPING ────────────────────────────────────────────────────────────
# Maps messy original column names → clean standard names

COLUMN_MAP = {
    "project name":                          "Project_Name",
    "project name ":                         "Project_Name",
    "gmpp id number":                        "GMPP_ID",
    "gmpp id":                               "GMPP_ID",
    "department":                            "Department",
    "annual report category":                "Category",
    "description / aims":                    "Description",
    "project description":                   "Description",
    "ipa delivery confidence assessment":    "IPA_RAG",
    "sro delivery confidence assessment":    "SRO_RAG",
    "project - start date":                  "Start_Date",
    "start date":                            "Start_Date",
    "project - end date":                    "End_Date",
    "end date":                              "End_Date",
    "departmental narrative on schedule, including any deviation from planned schedule (if necessary)": "Schedule_Narrative",
    "schedule narrative":                    "Schedule_Narrative",
    "financial year baseline (£m) (including non-government costs)": "FY_Baseline_GBPm",
    "financial year baseline (â£m) (including non-government costs)": "FY_Baseline_GBPm",
    "financial year baseline (â£m)":         "FY_Baseline_GBPm",
    "financial year forecast (£m) (including non-government costs)": "FY_Forecast_GBPm",
    "financial year forecast (â£m) (including non-government costs)": "FY_Forecast_GBPm",
    "financial year forecast (â£m)":         "FY_Forecast_GBPm",
    "financial year variance (%)":           "FY_Variance_Pct",
    "total baseline whole life costs (£m) (including non-government costs)": "WLC_Baseline_GBPm",
    "total baseline whole life costs (â£m) (including non-government costs)": "WLC_Baseline_GBPm",
    "whole life cost (â£m)":                 "WLC_Baseline_GBPm",
    "total baseline benefits (£m)":          "Benefits_GBPm",
    "total baseline benefits (â£m)":         "Benefits_GBPm",
    "benefits (â£m)":                        "Benefits_GBPm",
}

FINAL_COLUMNS = [
    "Year", "GMPP_ID", "Project_Name", "Department", "Category",
    "IPA_RAG", "SRO_RAG", "Risk_Label",
    "Start_Date", "End_Date",
    "FY_Baseline_GBPm", "FY_Forecast_GBPm", "FY_Variance_Pct",
    "WLC_Baseline_GBPm", "Benefits_GBPm",
    "Schedule_Narrative", "Source_File",
]

# ── AUTO-DETECTION HELPERS ────────────────────────────────────────────────────

def extract_year(filename):
    """Finds a 4-digit year (20xx) anywhere in the filename. Returns the LAST match
    (so 'report_2018_to_2019' picks 2019, matching the more recent reporting point)."""
    matches = re.findall(r'20\d{2}', filename)
    if matches:
        return int(matches[-1])
    return None

def read_csv_robust(filepath):
    """Reads a CSV trying utf-8 first, falling back to latin-1."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)
    df.columns = [str(c).lstrip('\ufeff').lstrip('ï»¿') for c in df.columns]
    return df

def detect_format(df):
    """
    Wide format: each COLUMN is a project → far more columns than rows.
    Long format: each ROW is a project → far more rows than columns.
    """
    n_rows, n_cols = df.shape
    return "wide" if n_cols > n_rows else "long"

# ── NORMALISER ────────────────────────────────────────────────────────────────
# The exact-match COLUMN_MAP above only covers a few literal column-name variants.
# The financial columns in particular change their wording almost every year
# (e.g. "2016/17 TOTAL Baseline £m" vs "Financial Year Baseline (£m)"), so an
# exact-match dictionary alone will silently miss most years. The function below
# adds a keyword/regex fallback so financial columns are caught regardless of
# which year's wording was used.

def fuzzy_financial_match(clean_name):
    """Keyword-based fallback for financial columns whose exact wording
    changes every reporting year. Returns a target column name, or None."""
    n = clean_name.lower()

    # Never match narrative / commentary text columns, even if they mention
    # "forecast", "variance", etc. — those are free-text, not numbers.
    if 'narrative' in n or 'commentary' in n:
        return None

    if 'whole life' in n and ('baseline' in n or 'cost' in n):
        return 'WLC_Baseline_GBPm'
    if 'benefit' in n and ('£' in clean_name or 'million' in n or '(m)' in n):
        return 'Benefits_GBPm'
    if 'variance' in n and ('%' in clean_name or 'pct' in n or 'percent' in n):
        return 'FY_Variance_Pct'
    if 'baseline' in n and ('£' in clean_name or 'million' in n or '(m)' in n):
        return 'FY_Baseline_GBPm'
    if 'forecast' in n and ('£' in clean_name or 'million' in n or '(m)' in n):
        return 'FY_Forecast_GBPm'
    return None

def normalise_columns(df):
    rename = {}
    for col in df.columns:
        clean = str(col).replace('\n', ' ').replace('\r', ' ').replace('  ', ' ').strip().lower()
        clean_short = clean.split('(')[0].strip()
        if clean in COLUMN_MAP:
            rename[col] = COLUMN_MAP[clean]
        elif clean_short in COLUMN_MAP:
            rename[col] = COLUMN_MAP[clean_short]
        else:
            fuzzy = fuzzy_financial_match(str(col))
            if fuzzy:
                rename[col] = fuzzy
    return df.rename(columns=rename)

def clean_money(val):
    """Strips currency symbols, thousand separators, and any encoding
    mangling (e.g. '£', 'Â£') from a value and returns a clean float.
    Returns NaN for blanks, dashes, 'N/A', etc."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ('', '-', 'N/A', 'n/a', 'TBC', 'tbc'):
        return np.nan
    negative = s.startswith('(') and s.endswith(')')
    # Extract only digits, decimal point, and minus sign
    match = re.findall(r'-?\d[\d,]*\.?\d*', s)
    if not match:
        return np.nan
    num_str = match[0].replace(',', '')
    try:
        num = float(num_str)
        return -num if negative else num
    except ValueError:
        return np.nan

# ── FORMAT-SPECIFIC READERS ───────────────────────────────────────────────────

def process_long_format(df, year, source_file):
    df = normalise_columns(df)
    df["Year"] = year
    df["Source_File"] = source_file
    return df

def process_wide_format(df, year, source_file):
    """Transposes a wide-format file: columns (projects) become rows."""
    attr_col = df.columns[0]
    attributes = df[attr_col].tolist()

    records = []
    for proj_col in df.columns[1:]:
        record = {"Project_Name": proj_col}
        for i, attr in enumerate(attributes):
            if pd.notna(attr):
                record[str(attr).strip()] = df[proj_col].iloc[i]
        records.append(record)

    transposed = pd.DataFrame(records)
    transposed = normalise_columns(transposed)
    transposed["Year"] = year
    transposed["Source_File"] = source_file
    return transposed

# ── RAG STANDARDISATION ───────────────────────────────────────────────────────

def standardise_rag(val):
    if pd.isna(val):
        return "Unknown"
    v = str(val).strip().lower()
    if v == "red":
        return "Red"
    elif v in ["amber/red", "amber-red"]:
        return "Amber/Red"
    elif v == "amber":
        return "Amber"
    elif v in ["amber/green", "amber-green"]:
        return "Amber/Green"
    elif v == "green":
        return "Green"
    else:
        return "Unknown"

def rag_to_risk_label(val):
    mapping = {
        "Red": "High Risk", "Amber/Red": "High Risk",
        "Amber": "Medium Risk",
        "Amber/Green": "Low Risk", "Green": "Low Risk",
        "Unknown": np.nan,
    }
    return mapping.get(val, np.nan)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GMPP DATA CONSOLIDATION (auto-detect mode)")
    print("=" * 60)

    csv_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.csv")))
    if not csv_files:
        print(f"\n❌ No CSV files found in: {os.path.abspath(DATA_FOLDER)}")
        print("   Make sure this script is in the same folder as your CSV files.")
        return

    print(f"\nFound {len(csv_files)} CSV file(s):\n")

    all_dfs = []
    for filepath in csv_files:
        filename = os.path.basename(filepath)

        if filename in MANUAL_YEAR_OVERRIDE:
            year = MANUAL_YEAR_OVERRIDE[filename]
        else:
            year = extract_year(filename)

        if year is None:
            print(f"  ⚠️  SKIPPED (no year found in name): {filename}")
            continue

        try:
            df = read_csv_robust(filepath)
        except Exception as e:
            print(f"  ⚠️  SKIPPED (could not read): {filename} — {e}")
            continue

        fmt = detect_format(df)
        try:
            if fmt == "wide":
                processed = process_wide_format(df, year, filename)
            else:
                processed = process_long_format(df, year, filename)
        except Exception as e:
            print(f"  ⚠️  SKIPPED (processing error): {filename} — {e}")
            continue

        print(f"  ✅ {year} [{fmt:4s}] {filename}  →  {len(processed)} projects")
        all_dfs.append(processed)

    if not all_dfs:
        print("\n❌ No files were successfully processed.")
        return

    print("\n[Combining all years...]")
    all_dfs = [df.loc[:, ~df.columns.duplicated()] for df in all_dfs]
    combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    print(f"  Total records before filtering: {len(combined)}")

    if "IPA_RAG" in combined.columns:
        combined["IPA_RAG"] = combined["IPA_RAG"].apply(standardise_rag)
        combined["Risk_Label"] = combined["IPA_RAG"].apply(rag_to_risk_label)

    # Clean financial columns: strip £, commas, encoding artefacts -> proper float
    money_cols = ["FY_Baseline_GBPm", "FY_Forecast_GBPm", "FY_Variance_Pct",
                  "WLC_Baseline_GBPm", "Benefits_GBPm"]
    for col in money_cols:
        if col in combined.columns:
            combined[col] = combined[col].apply(clean_money)

    keep_cols = [c for c in FINAL_COLUMNS if c in combined.columns]
    combined_clean = combined[keep_cols].copy()

    combined_clean = combined_clean[combined_clean["Project_Name"].notna()]
    combined_clean["Project_Name"] = combined_clean["Project_Name"].astype(str).str.strip()
    combined_clean = combined_clean[combined_clean["Project_Name"] != ""]

    out_full = os.path.join(DATA_FOLDER, "GMPP_Consolidated.csv")
    combined_clean.to_csv(out_full, index=False, encoding='utf-8-sig')
    print(f"\n✅ Full dataset saved: {out_full}  ({len(combined_clean)} rows)")

    print("\n── Years Covered ──")
    print(sorted(combined_clean["Year"].unique()))

    if "Category" in combined_clean.columns:
        construction_mask = combined_clean["Category"].str.contains(
            "Infrastructure|Construction", case=False, na=False
        )
        construction_df = combined_clean[construction_mask].copy()

        out_construction = os.path.join(DATA_FOLDER, "GMPP_Construction.csv")
        construction_df.to_csv(out_construction, index=False, encoding='utf-8-sig')
        print(f"✅ Construction-only dataset saved: {out_construction}  ({len(construction_df)} rows)")

        print("\n── Construction Projects by Year ──")
        if "Year" in construction_df.columns:
            print(construction_df.groupby("Year").size().to_string())

        if "Risk_Label" in construction_df.columns:
            print("\n── Risk Label Distribution (Construction) ──")
            print(construction_df["Risk_Label"].value_counts(dropna=False).to_string())
    else:
        print("\n⚠️  Category column not found in any file — skipping construction filter.")
        print("   (This happens if none of your files are in 'long' format, since only")
        print("   long-format files contain the Annual Report Category column.)")

    print("\n── Done. Import GMPP_Construction.csv into Power BI. ──")

if __name__ == "__main__":
    main()
