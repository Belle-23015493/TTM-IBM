from pathlib import Path
import pandas as pd

# ========================================================= #
# GLOBAL CONFIGURATION
# ========================================================= #
# This section defines where your project and datasets live.
# Using Path makes the code portable across computers.

# Get project root (one level above this script)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder containing all datasets
BASE_DIR = PROJECT_ROOT / "datasets"

# Raw PSI dataset
PSI_FILE = BASE_DIR / "historical-24-hr-psi_010414-311219.csv"

# Final output file (ONE clean dataset only)
CLEAN_OUT = BASE_DIR / "clean_half_hourly_dataset.csv"


# ========================================================= #
# 1) HALF-HOURLY SYSTEM DEMAND CLEANING
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
    metrics = ["System Demand (Actual)"]

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
# 2) PSI CLEANING
# ========================================================= #
def clean_psi(psi_csv_path: str) -> pd.DataFrame:
    print(f"[PSI] Reading PSI file: {psi_csv_path}")
    psi = pd.read_csv(psi_csv_path)

    # 1) Rename datetime column
    if "24-hr_psi" not in psi.columns:
        raise ValueError("Expected a '24-hr_psi' column in the PSI CSV.")
    psi = psi.rename(columns={"24-hr_psi": "datetime_str"})

    # 2) Parse datetime strings
    ts = pd.to_datetime(
        psi["datetime_str"],
        dayfirst=True,
        errors="coerce"
    )
    valid_mask = ts.notna()
    if not valid_mask.all():
        print(f"[PSI] Dropping {(~valid_mask).sum()} rows with invalid datetime.")
    psi = psi.loc[valid_mask].copy()
    ts = ts[valid_mask]

    # 3) Convert period-ending hourly timestamps -> period-start
    # Shift all timestamps back by 1 hour to align with period-start
    psi["Timestamp"] = ts - pd.Timedelta(hours=1)

    # --- ADDED STEP: Remove years 2014, 2015, 2016 ---
    psi = psi[~psi["Timestamp"].dt.year.isin([2014, 2015, 2016])]
    print(f"[PSI] After filtering 2014-2016, shape: {psi.shape}")

    # 4) Rename PSI region columns
    rename_map = {
        "north": "PSI_North",
        "south": "PSI_South",
        "east": "PSI_East",
        "west": "PSI_West",
        "central": "PSI_Central",
    }
    psi = psi.rename(columns=rename_map)
    psi_cols = [c for c in rename_map.values() if c in psi.columns]

    # 5) Sort & remove fully missing rows
    psi = psi.sort_values("Timestamp")
    psi = psi.dropna(how="all", subset=psi_cols)

    # 6) Handle duplicates safely
    psi = psi.groupby("Timestamp", as_index=False).mean(numeric_only=True)

    # 7) Resample hourly -> 30 minutes
    psi = psi.set_index("Timestamp")
    # Note: 30T is deprecated in newer pandas, use 30min or 30min
    psi_30 = (
        psi
        .resample("30min")
        .interpolate(method="time")
        .reset_index()
    )

    return psi_30


# ========================================================= #
# 3) TEMPERATURE CLEANING (2017–2019)
# ========================================================= #
def clean_temperature(base_dir: Path) -> pd.DataFrame:
    """
    Cleans NEA temperature data and aggregates to 30-minute intervals.

    Processing steps:
    1. Load all yearly CSVs.
    2. Keep timestamp, station_id, and temperature reading.
    3. Remove missing or extreme temperature values.
    4. Convert timestamps to timezone-naive for consistency.
    5. Average across stations per timestamp (Singapore-wide).
    6. Floor timestamps to 30-minute bins and average within each bin.

    Final output:
        Timestamp | Temp_C_30min
    """
    print("[TEMP] Cleaning temperature data")

    # --- 1. Load all years ---
    files = sorted(base_dir.glob("HistoricalAirTemperatureacrossSingapore*.csv"))
    if not files:
        raise FileNotFoundError("No temperature CSV files found in datasets folder.")

    dfs = [pd.read_csv(f) for f in files]
    temp_all = pd.concat(dfs, ignore_index=True)

    # --- 2. Keep relevant columns ---
    temp_all = temp_all[["timestamp", "station_id", "reading_value"]]
    temp_all = temp_all.rename(columns={"reading_value": "temp_c"})

    # --- 3. Parse timestamp and remove timezone ---
    temp_all["timestamp"] = pd.to_datetime(temp_all["timestamp"]).dt.tz_localize(None)

    # --- 4. Remove missing or extreme values ---
    temp_all = temp_all.dropna(subset=["temp_c"])
    temp_all = temp_all[(temp_all["temp_c"] >= 20) & (temp_all["temp_c"] <= 40)]

    # --- 5. Average across stations per timestamp ---
    temp_mean = temp_all.groupby("timestamp", as_index=False)["temp_c"].mean()

    # --- 6. Convert to 30-minute intervals ---
    temp_mean["Timestamp"] = temp_mean["timestamp"].dt.round("30min")
    temp_30 = temp_mean.groupby("Timestamp", as_index=False)["temp_c"].mean()
    temp_30 = temp_30.rename(columns={"temp_c": "Temp"})

    return temp_30


# ========================================================= #
# 4) BUILD FINAL CLEAN DATASET
# ========================================================= #
def build_clean_dataset(base_dir: Path) -> pd.DataFrame:
    """
    Merges all cleaned datasets into ONE final dataset.
    """

    print("\n=== BUILDING CLEAN DATASET ===")

    # --- Load & clean individual datasets ---
    demand = clean_half_hourly_demand(base_dir)
    psi = clean_psi(PSI_FILE)
    temp = clean_temperature(base_dir)

    # --- MERGE DATASETS ---
    # Inner join ensures all datasets align perfectly in time
    clean_df = (
        demand
        .merge(psi, on="Timestamp", how="inner")
        .merge(temp, on="Timestamp", how="inner")
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    return clean_df


# ========================================================= #
# 5) MAIN SCRIPT
# ========================================================= #
if __name__ == "__main__":
    clean_dataset = build_clean_dataset(BASE_DIR)

    # Save final clean dataset
    clean_dataset.to_csv(CLEAN_OUT, index=False)

    print("\nClean dataset saved to:")
    print(CLEAN_OUT)
    print("\nPreview:")
    print(clean_dataset.head())