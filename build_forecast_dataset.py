from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(
    r"C:\Users\belle\OneDrive - Republic Polytechnic\Desktop\Curriculum\Year 3 Sem 2\FYP\python\datasets"
)

MASTER_FILE = BASE_DIR / "master_half_hourly_dataset.csv"
FORECAST_OUT = BASE_DIR / "forecast_dataset.csv"


def build_forecast_dataset() -> pd.DataFrame:
    print("Reading master dataset...")
    df = pd.read_csv(MASTER_FILE, parse_dates=["Timestamp"])

    # 1. Drop columns not needed for forecasting
    cols_to_drop = [
        "NEM Demand (Actual)",
        "NEM Demand (Forecast)",
        "INFORMATION TYPE",
        "USEP_DEMAND (MW)",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # 2. Rename System Demand column to something clearer
    #    Original: "System Demand (Actual)"
    #    New:      "System Demand"
    df = df.rename(columns={"System Demand (Actual)" : "System_Demand", "PRICE ($/MWh)" : "USEP_Price_MWh"})

    return df


if __name__ == "__main__":
    print("Building forecasting dataset from master file...")
    forecast_df = build_forecast_dataset()

    print("\nPreview of forecasting dataset:")
    print(forecast_df.head())

    forecast_df.to_csv(FORECAST_OUT, index=False)
    print("\nSaved forecasting dataset to:")
    print(f"  {FORECAST_OUT}")