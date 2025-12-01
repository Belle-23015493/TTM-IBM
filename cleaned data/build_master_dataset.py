from pathlib import Path
import pandas as pd

# ===================== GLOBAL CONFIG ===================== #
# Automatically find the project root and datasets folder
# This file lives in: project/cleaned data/build_master_dataset.py
# So project root is one level up from this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "datasets"

# Raw PSI file (inside datasets/)
PSI_FILE = BASE_DIR / "historical-24-hr-psi_010414-311219.csv"

# Final single joined output (only one file!), also saved in datasets/
MASTER_OUT = BASE_DIR / "master_half_hourly_dataset.csv"
# ========================================================= #


# ========================================================= #
# 1) HALF-HOURLY-DEMAND CLEANING
# ========================================================= #

def clean_half_hourly_demand(base_dir: Path) -> pd.DataFrame:
    """
    Clean the half-hourly-demand-2015-2020 dataset.

    For each weekly file:
      - Start from row 6 (no headers).
      - Name columns as Time, Mon_System_Demand, ..., Sun_NEM_Forecast.
      - Unpivot, split Day/Metric, pivot back to 3 metric columns.
      - Use filename (YYYYMMDD) as Monday date and derive Tue–Sun dates.
      - Create a single Timestamp column: Date + Time.

    Returns a big DataFrame with:
      Timestamp, System_Demand, NEM_Demand, NEM_Forecast
    covering all weeks.
    """

    # Folder structure: base_dir / half-hourly-demand-2015-2020 / half-hourly-demand / <year> / *.xls
    root = base_dir / "half-hourly-demand-2015-2020" / "half-hourly-demand"
    excel_files = sorted(root.glob("*/*.xls"))

    if not excel_files:
        raise FileNotFoundError(f"No .xls files found under {root}")

    all_weeks: list[pd.DataFrame] = []

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    metrics = [
    "System Demand (Actual)",
    "NEM Demand (Actual)",
    "NEM Demand (Forecast)"
    ]

    # Build the expected column names: Time + 7*3 metric columns
    expected_cols = ["Time"]
    for d in days:
        for m in metrics:
            expected_cols.append(f"{d}_{m}")

    n_expected = len(expected_cols)  # should be 22

    # Map day code -> offset (days after Monday)
    day_offset = {d: i for i, d in enumerate(days)}

    for path in excel_files:
        print(f"[Half-hourly] Processing {path.name} ...")

        # Read with NO header so we can control everything
        raw = pd.read_excel(path, header=None)

        # ---- Loop 1: start from row 6 (index 5) without header ----
        # Take only the first n_expected columns to match our header scheme
        data = raw.iloc[5:, :n_expected].copy()

        # If the file doesn't have enough columns, skip it
        if data.shape[1] < n_expected:
            print(
                f"  Warning: {path.name} has {data.shape[1]} columns, "
                f"expected {n_expected}. Skipping."
            )
            continue

        # Assign our own header: Time, Mon_System_Demand, ..., Sun_NEM_Forecast
        data.columns = expected_cols

        # Clean Time column and drop empty time rows
        data["Time"] = data["Time"].astype(str).str.strip()
        data = data[~data["Time"].isin(["", "nan", "NaT"])]

        # Melt all day/metric columns into long format
        long_df = data.melt(
            id_vars=["Time"],
            var_name="Day_Metric",
            value_name="value",
        )

        # Split "Mon_System_Demand" -> "Mon" + "System_Demand"
        day_metric = long_df["Day_Metric"].str.split("_", n=1, expand=True)
        long_df["Day"] = day_metric[0]
        long_df["Metric"] = day_metric[1]

        # Drop rows where Day or Metric is missing
        long_df = long_df.dropna(subset=["Day", "Metric"])

        # Pivot back so we get one row per (Day, Time) with columns:
        # System_Demand, NEM_Demand, NEM_Forecast
        week = (
            long_df.pivot_table(
                index=["Day", "Time"],
                columns="Metric",
                values="value",
                aggfunc="first",
            )
            .reset_index()
        )

        # Flatten columns (pivot_table puts Metric names in a column MultiIndex)
        week.columns = ["Day", "Time"] + [str(c) for c in week.columns[2:]]

        # ---- Loop 2: derive Date from filename (Monday) ----
        # e.g. "20150105.xls" -> "20150105" -> 2015-01-05
        monday_str = path.stem  # filename without .xls
        monday_date = pd.to_datetime(monday_str, format="%Y%m%d")

        # Map each row's Day to an integer offset from Monday (0..6)
        week["Day_offset"] = week["Day"].map(day_offset)

        # Remove any weird rows where Day isn't Mon..Sun
        week = week.dropna(subset=["Day_offset"])

        # Create true Date = Monday + offset days
        week["Date"] = week["Day_offset"].astype(int).apply(
            lambda d: monday_date + pd.Timedelta(days=d)
        )

        # Combine Date + Time -> Timestamp
        week["Timestamp"] = pd.to_datetime(
            week["Date"].dt.strftime("%Y-%m-%d") + " " + week["Time"].astype(str),
            errors="coerce",
        )

        # Ensure metric columns are numeric
        for col in metrics:
            if col in week.columns:
                week[col] = pd.to_numeric(week[col], errors="coerce")

        # Keep only the columns we care about
        keep_cols = ["Timestamp"] + [c for c in metrics if c in week.columns]
        week = week[keep_cols]

        # Drop rows with no Timestamp
        week = week.dropna(subset=["Timestamp"])

        all_weeks.append(week)

    if not all_weeks:
        raise ValueError("No valid weekly files were processed (all_weeks is empty).")

    # ---- Combine all weeks ----
    hh = pd.concat(all_weeks, ignore_index=True)
    hh = hh.sort_values("Timestamp").reset_index(drop=True)

    return hh


# ========================================================= #
# 2) USEP CLEANING
# ========================================================= #

def period_to_time(period) -> str | None:
    """
    Map USEP period (1..48) to a Period Ending Time string consistent
    with your half-hourly-demand dataset.

    Period 1  -> 00:30  (covers 00:00-00:30)
    Period 2  -> 01:00  (covers 00:30-01:00)
    ...
    Period 47 -> 23:30
    Period 48 -> 00:00  (end of the day, same calendar DATE)
    """
    try:
        p = int(period)
    except (TypeError, ValueError):
        return None

    if p < 1 or p > 48:
        return None

    minutes = p * 30  # 1->30, 2->60, ..., 48->1440
    if minutes == 1440:
        return "00:00"  # 24:00 shown as 00:00

    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def find_usep_root(base_dir: Path) -> Path:
    """
    Locate the USEP root folder.

    Expected structure:
        base_dir / "usep-2015-2020" / "usep-raw" / <year> / *.csv
    """
    candidate1 = base_dir / "usep-2015-2020" / "usep-raw"   # <-- your real path
    candidate2 = base_dir / "usep-2015-2020"                # just in case
    candidate3 = base_dir / "usep"                          # fallback

    for c in (candidate1, candidate2, candidate3):
        if c.is_dir():
            return c

    raise FileNotFoundError(
        "Could not find USEP folder. "
        "Expected something like 'datasets/usep-2015-2020/usep-raw/'."
    )


def clean_usep(base_dir: Path) -> pd.DataFrame:
    """
    Clean the USEP dataset (CSV-only) and return a single DataFrame.
    """
    root = find_usep_root(base_dir)
    print(f"[USEP] Root folder: {root}")

    # Collect only CSV files in all subfolders
    files = [f for f in root.rglob("*.csv") if not f.name.startswith("~$")]

    if not files:
        raise FileNotFoundError(f"No USEP CSV files found under {root}")

    all_frames: list[pd.DataFrame] = []

    for path in sorted(files):
        print(f"[USEP] Processing {path.relative_to(root)} ...")

        df = pd.read_csv(path)

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        required_basic = {"INFORMATION TYPE", "DATE", "PERIOD"}
        if not required_basic.issubset(df.columns):
            print(
                f"  Warning: missing one of {required_basic} in {path.name}, skipping."
            )
            continue

        # ---- Handle price column ----
        # If it’s called USEP ($/MWh), rename to PRICE ($/MWh)
        if "USEP ($/MWh)" in df.columns:
            df = df.rename(columns={"USEP ($/MWh)": "PRICE ($/MWh)"})

        if "PRICE ($/MWh)" not in df.columns:
            # maybe different naming like 'Usep ($/MWh)' etc.
            usep_like = [
                c
                for c in df.columns
                if "USEP" in c.upper() and "/MWH" in c.upper()
            ]
            if usep_like:
                df = df.rename(columns={usep_like[0]: "PRICE ($/MWh)"})
            else:
                print(f"  Warning: no USEP/PRICE column in {path.name}, skipping.")
                continue

        # ---- Drop LCP and TCL if present ----
        for drop_col in ["LCP ($/MWh)", "TCL (MW)"]:
            if drop_col in df.columns:
                df = df.drop(columns=[drop_col])

        # ---- Find DEMAND column ----
        demand_col = None
        for c in df.columns:
            if "DEMAND" in c.upper():
                demand_col = c
                break

        if demand_col is None:
            print(
                f"  Warning: no DEMAND column in {path.name}, setting USEP_DEMAND (MW) as NaN."
            )
            df["USEP_DEMAND (MW)"] = pd.NA
        else:
            # Rename whatever demand-like column to USEP_DEMAND (MW)
            df = df.rename(columns={demand_col: "USEP_DEMAND (MW)"})

        # Keep the columns we care about
        df = df[
            ["INFORMATION TYPE", "DATE", "PERIOD", "PRICE ($/MWh)", "USEP_DEMAND (MW)"]
        ].copy()

        # ---- Parse DATE ----
        df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")

        # ---- PERIOD -> Time (Period Ending Time) ----
        df["Time"] = df["PERIOD"].apply(period_to_time)

        df = df.dropna(subset=["DATE", "Time"])

        # ---- Build Timestamp ----
        df["Timestamp"] = pd.to_datetime(
            df["DATE"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str),
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )
        df = df.dropna(subset=["Timestamp"])

        # ---- Numeric types ----
        df["PRICE ($/MWh)"] = pd.to_numeric(df["PRICE ($/MWh)"], errors="coerce")
        df["USEP_DEMAND (MW)"] = pd.to_numeric(df["USEP_DEMAND (MW)"], errors="coerce")

        # Final output columns (drop DATE, PERIOD, Time)
        df = df[["Timestamp", "INFORMATION TYPE", "PRICE ($/MWh)", "USEP_DEMAND (MW)"]]

        all_frames.append(df)

    if not all_frames:
        raise ValueError("No valid USEP rows produced - please check your CSV files.")

    usep = pd.concat(all_frames, ignore_index=True)
    usep = usep.sort_values("Timestamp").reset_index(drop=True)
    return usep


# ========================================================= #
# 3) PSI CLEANING
# ========================================================= #

def clean_psi(psi_csv_path: Path) -> pd.DataFrame:
    print(f"[PSI] Reading PSI file: {psi_csv_path}")
    psi = pd.read_csv(psi_csv_path)

    # --- 3.1 Rename first column to a clearer name ---------------------------
    # In the raw file this is usually "24-hr_psi" and contains datetime strings.
    if "24-hr_psi" not in psi.columns:
        raise ValueError("Expected a '24-hr_psi' column in the PSI CSV.")
    psi = psi.rename(columns={"24-hr_psi": "datetime_str"})

    # --- 3.2 Fix malformed times like "... 010:00:00" ------------------------
    psi["datetime_str"] = (
        psi["datetime_str"]
        .astype(str)
        .str.replace(" 010:00:00", " 00:00:00")
    )

    # --- 3.3 Parse the date/time strings -------------------------------------
    ts = pd.to_datetime(psi["datetime_str"], dayfirst=True, errors="coerce")

    # Drop any rows that still failed to parse
    valid_mask = ts.notna()
    if not valid_mask.all():
        print(f"[PSI] Dropping {(~valid_mask).sum()} rows with invalid datetime.")
    psi = psi.loc[valid_mask].copy()
    ts = ts[valid_mask]

    # --- 3.4 Standardise 00:00 logic -----------------------------------------
    # PSI file treats 00:00 as NEXT day; half-hourly & USEP treat 00:00
    # as belonging to the SAME day as the previous readings.
    mask_midnight = ts.dt.hour == 0
    ts_adjusted = ts.where(~mask_midnight, ts - pd.Timedelta(days=1))

    psi["Timestamp"] = ts_adjusted

    # We don't need the original string column anymore
    psi = psi.drop(columns=["datetime_str"])

    # --- 3.5 Rename PSI region columns for clarity ----------------------------
    rename_map = {
        "north": "PSI_North",
        "south": "PSI_South",
        "east": "PSI_East",
        "west": "PSI_West",
        "central": "PSI_Central",
    }
    psi = psi.rename(columns=rename_map)

    # Determine which PSI columns actually exist in this file
    psi_cols = [c for c in ["PSI_North", "PSI_South", "PSI_East", "PSI_West", "PSI_Central"] if c in psi.columns]
    if not psi_cols:
        raise ValueError("No PSI region columns (north/south/east/west/central) found after renaming.")

    # --- 3.6 Sort & handle duplicates ----------------------------------------
    psi = psi.sort_values("Timestamp")

    # Drop rows where ALL PSI values are NaN (just in case)
    psi = psi.dropna(how="all", subset=psi_cols)

    # If there are duplicate timestamps, average numeric columns
    psi = psi.groupby("Timestamp", as_index=False).mean(numeric_only=True)

    # --- 3.7 Resample from hourly → 30 minutes -------------------------------
    psi = psi.set_index("Timestamp")

    # Create 30-minute steps and interpolate between the hourly points
    psi_30 = psi.resample("30T").interpolate(method="time")

    # Bring Timestamp back as a normal column
    psi_30 = psi_30.reset_index()

    return psi_30


# ========================================================= #
# 4) BUILD MASTER DATASET (JOIN ALL THREE)
# ========================================================= #

def build_master_dataset(base_dir: Path) -> pd.DataFrame:
    print("\n=== STEP 1: Clean half-hourly demand ===")
    hh = clean_half_hourly_demand(base_dir)
    print(f"  Half-hourly rows: {len(hh)}")

    print("\n=== STEP 2: Clean USEP ===")
    usep = clean_usep(base_dir)
    print(f"  USEP rows: {len(usep)}")

    print("\n=== STEP 3: Clean PSI ===")
    psi = clean_psi(PSI_FILE)
    print(f"  PSI (30-min) rows: {len(psi)}")

    # Ensure all have Timestamp as datetime (should already be ok)
    hh["Timestamp"] = pd.to_datetime(hh["Timestamp"])
    usep["Timestamp"] = pd.to_datetime(usep["Timestamp"])
    psi["Timestamp"] = pd.to_datetime(psi["Timestamp"])

    print("\n=== STEP 4: Join all three on Timestamp (inner join) ===")
    # First join half-hourly with USEP
    df = hh.merge(usep, on="Timestamp", how="inner")

    # Then join with PSI
    df = df.merge(psi, on="Timestamp", how="inner")

    # Sort by Timestamp
    df = df.sort_values("Timestamp").reset_index(drop=True)

    print(f"  Final merged rows: {len(df)}")
    return df


# ========================================================= #
# 5) MAIN
# ========================================================= #

if __name__ == "__main__":
    print("Building master half-hourly dataset from ALL three sources...\n")
    master = build_master_dataset(BASE_DIR)

    print("\nPreview of final dataset:")
    print(master.head())

    master.to_csv(MASTER_OUT, index=False)
    print(f"  {MASTER_OUT}")