"""
GMPP Delay Label Builder
=========================
Builds the binary delay label that A1 specified, adapted to the ACTUAL
structure of the consolidated GMPP_Construction.csv: it is panel data
(the same project appears in multiple Year rows as its annual snapshot),
not one row per project with a single baseline/forecast date pair.

Delay is therefore computed at the PROJECT level by comparing each
project's End_Date as first reported (earliest Year available) against
its End_Date as most recently reported (latest Year available):

    Slippage_Days = Latest_End_Date - Baseline_End_Date
    Delayed       = 1 if Slippage_Days > 0, else 0

For projects with only one valid End_Date observation across all years
(no second snapshot to compare against), a fallback proxy is used based
on the IPA Delivery Confidence Assessment (IPA_RAG) at that snapshot:
    Red / Amber-Red     -> Delayed = 1
    Green / Amber-Green -> Delayed = 0
    Amber / Unknown     -> Delayed = NaN (left unresolved, not guessed)

Every row carries a Label_Method column (parallel to the existing
Classification_Method column) documenting which of the two approaches
produced its label, so this can be reported transparently as a
methodological limitation.

Inputs:
    GMPP_Construction.csv   (the 525-row file already produced by
                              gmpp_consolidate.py)

Outputs:
    GMPP_Construction_Labeled.csv  — original 525 rows + new columns
                                     (End_Date_Parsed, Slippage_Days,
                                     Delayed, Label_Method), for Power BI
                                     trend/snapshot visuals
    GMPP_Project_Outcome.csv      — one row per project (250 rows),
                                     baseline features + final label,
                                     for ML training and for the Power BI
                                     project-outcome / risk-tier table
"""

import pandas as pd
import numpy as np
import os

DATA_FOLDER = "."
INPUT_FILE = os.path.join(DATA_FOLDER, "GMPP_Construction.csv")

RAG_DELAY_PROXY = {
    "Red": 1, "Amber/Red": 1,
    "Green": 0, "Amber/Green": 0,
    "Amber": np.nan, "Unknown": np.nan,
}

# Casing-only duplicates observed across annual releases (same department,
# inconsistent abbreviation casing between report years) -> merge.
# Genuine machinery-of-government changes (DECC->BEIS->DESNZ, DoH->DHSC,
# DfID->FCDO) are NOT merged here, since they reflect real departmental
# history rather than data entry inconsistency.
DEPARTMENT_CASING_FIX = {
    "DfT": "DFT", "DFT": "DFT",
    "DfE": "DFE", "DFE": "DFE",
    "DfID": "DFID", "DFID": "DFID",
    "MoD": "MOD", "MOD": "MOD",
    "MoJ": "MOJ", "MOJ": "MOJ",
    "DoH Capital": "DoH",
}


def clean_department(d):
    if pd.isna(d):
        return d
    return DEPARTMENT_CASING_FIX.get(str(d).strip(), str(d).strip())


def build_project_name_lookup(df):
    """Several GMPP annual releases render some project names in ALL CAPS
    while other years use standard title case for the SAME project (e.g.
    'Lower Thames Crossing' for 2017-2023, then 'LOWER THAMES CROSSING' for
    2024 only). Grouping on the raw string therefore fractures a single
    project's history into two artificial 'projects', each with a shorter
    and incorrect baseline/latest comparison. This builds a normalised key
    (stripped, lower-cased) for grouping, and a canonical display name
    (the most frequently occurring casing variant, breaking ties with the
    most recent year) for reporting."""
    tmp = df.copy()
    tmp["Project_Key"] = tmp["Project_Name"].str.strip().str.lower()
    canonical = (
        tmp.groupby(["Project_Key", "Project_Name"])["Year"].agg(count="count", max_year="max")
        .reset_index()
        .sort_values(["Project_Key", "count", "max_year"], ascending=[True, False, False])
        .groupby("Project_Key")
        .first()["Project_Name"]
    )
    return tmp["Project_Key"].values, tmp["Project_Key"].map(canonical).values


def parse_date(s):
    """Parses End_Date/Start_Date across the mixed formats observed in
    GMPP annual releases (DD/MM/YYYY, DD/MM/YY, YYYY-MM-DD) and treats
    FOI-exemption text, 'Not provided', etc. as missing rather than a
    parse error."""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    if s == "" or s.lower() in ("not provided", "n/a", "tbc", "unknown"):
        return pd.NaT
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT  # FOI-exemption narratives and anything else unparseable


def main():
    df = pd.read_csv(INPUT_FILE)
    df["Department"] = df["Department"].apply(clean_department)
    df["End_Date_Parsed"] = df["End_Date"].apply(parse_date)
    df["Start_Date_Parsed"] = df["Start_Date"].apply(parse_date)
    df["Project_Key"], df["Project_Name_Canonical"] = build_project_name_lookup(df)

    n_unparsed = df["End_Date_Parsed"].isna().sum()
    print(f"Rows with unparseable End_Date (FOI-exempt / not provided / etc.): {n_unparsed} / {len(df)}")
    print(f"Distinct projects by raw Project_Name string: {df['Project_Name'].nunique()}")
    print(f"Distinct projects after case/whitespace normalisation: {df['Project_Key'].nunique()}")

    # ---- Project-level baseline vs latest comparison (grouped on the
    #      normalised key, so a project recorded in ALL CAPS in one year
    #      and title case in another is treated as the single project it
    #      actually is) ----
    valid = df.dropna(subset=["End_Date_Parsed"]).sort_values(["Project_Key", "Year"])
    grouped = valid.groupby("Project_Key")

    project_records = []
    for key, g in grouped:
        g = g.sort_values("Year")
        baseline = g.iloc[0]
        latest = g.iloc[-1]
        n_years = g["Year"].nunique()

        record = {
            "Project_Key": key,
            "Project_Name": baseline["Project_Name_Canonical"],
            "Department": baseline["Department"],
            "Classification_Method": baseline["Classification_Method"],
            "Baseline_Year": int(baseline["Year"]),
            "Baseline_Start_Date": baseline["Start_Date_Parsed"].date() if pd.notna(baseline["Start_Date_Parsed"]) else np.nan,
            "Baseline_End_Date": baseline["End_Date_Parsed"].date(),
            "Planned_Duration_Days": (baseline["End_Date_Parsed"] - baseline["Start_Date_Parsed"]).days if pd.notna(baseline["Start_Date_Parsed"]) else np.nan,
            "Baseline_IPA_RAG": baseline["IPA_RAG"],
            "Baseline_WLC_GBPm": baseline["WLC_Baseline_GBPm"],
            "Baseline_FY_Cost_GBPm": baseline["FY_Baseline_GBPm"],
            "Latest_Year": int(latest["Year"]),
            "Latest_End_Date": latest["End_Date_Parsed"].date(),
            "Latest_IPA_RAG": latest["IPA_RAG"],
            "N_Years_Observed": int(n_years),
        }

        if n_years >= 2:
            slip = (latest["End_Date_Parsed"] - baseline["End_Date_Parsed"]).days
            record["Slippage_Days"] = slip
            record["Delayed"] = int(slip > 0)
            record["Label_Method"] = "Schedule Slippage (>=2 valid annual observations)"
        else:
            proxy = RAG_DELAY_PROXY.get(baseline["IPA_RAG"], np.nan)
            record["Slippage_Days"] = np.nan
            record["Delayed"] = proxy
            record["Label_Method"] = (
                "RAG Proxy (single valid observation, no slippage comparison possible)"
                if not pd.isna(proxy)
                else "Unresolved (single observation, Amber/Unknown RAG)"
            )
        project_records.append(record)

    # ---- Projects with NO valid End_Date in any year at all ----
    no_date_keys = set(df["Project_Key"].unique()) - set(p["Project_Key"] for p in project_records)
    for key in no_date_keys:
        sub = df[df["Project_Key"] == key].sort_values("Year")
        latest = sub.iloc[-1]
        proxy = RAG_DELAY_PROXY.get(latest["IPA_RAG"], np.nan)
        project_records.append({
            "Project_Key": key,
            "Project_Name": latest["Project_Name_Canonical"],
            "Department": latest["Department"],
            "Classification_Method": latest["Classification_Method"],
            "Baseline_Year": int(sub.iloc[0]["Year"]),
            "Baseline_Start_Date": sub.iloc[0]["Start_Date_Parsed"].date() if pd.notna(sub.iloc[0]["Start_Date_Parsed"]) else np.nan,
            "Baseline_End_Date": np.nan,
            "Planned_Duration_Days": np.nan,
            "Baseline_IPA_RAG": sub.iloc[0]["IPA_RAG"],
            "Baseline_WLC_GBPm": sub.iloc[0]["WLC_Baseline_GBPm"],
            "Baseline_FY_Cost_GBPm": sub.iloc[0]["FY_Baseline_GBPm"],
            "Latest_Year": int(latest["Year"]),
            "Latest_End_Date": np.nan,
            "Latest_IPA_RAG": latest["IPA_RAG"],
            "N_Years_Observed": int(sub["Year"].nunique()),
            "Slippage_Days": np.nan,
            "Delayed": proxy,
            "Label_Method": (
                "RAG Proxy (no valid End_Date in any year)"
                if not pd.isna(proxy)
                else "Unresolved (no valid End_Date, Amber/Unknown RAG)"
            ),
        })

    project_outcome = pd.DataFrame(project_records).sort_values("Project_Name")

    # ---- Broadcast project-level label back onto every snapshot row ----
    label_lookup = project_outcome.set_index("Project_Key")[["Slippage_Days", "Delayed", "Label_Method"]]
    df = df.join(label_lookup, on="Project_Key", rsuffix="")

    df.to_csv(os.path.join(DATA_FOLDER, "GMPP_Construction_Labeled.csv"), index=False, encoding="utf-8-sig")
    project_outcome.to_csv(os.path.join(DATA_FOLDER, "GMPP_Project_Outcome.csv"), index=False, encoding="utf-8-sig")

    # ---- Summary ----
    print(f"\nDistinct projects (deduplicated): {len(project_outcome)}")
    print("\nLabel_Method distribution (project level):")
    print(project_outcome["Label_Method"].value_counts().to_string())
    print(f"\nResolved labels (project level): {project_outcome['Delayed'].notna().sum()} / {len(project_outcome)}")
    print("\nDelayed distribution (project level, resolved only):")
    print(project_outcome["Delayed"].value_counts(dropna=True).to_string())
    print(f"\nResolved labels (row level, 525 rows): {df['Delayed'].notna().sum()} / {len(df)}")
    print("\nOutputs written:")
    print("  - GMPP_Construction_Labeled.csv  (row-level, for trend visuals)")
    print("  - GMPP_Project_Outcome.csv       (project-level, for ML + outcome dashboard)")


if __name__ == "__main__":
    main()
