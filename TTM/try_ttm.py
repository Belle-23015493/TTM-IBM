# =============================================================
# TinyTimeMixer Forecast — FYP Final Version with Monthly Plot
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# =============================================================
# 1. LOAD DATA
# =============================================================

# This file lives in: project/TTM/ttm_model.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "datasets" / "forecast_dataset.csv"

print(f"Loading data from: {DATA_FILE}")
df = pd.read_csv(DATA_FILE, parse_dates=["Timestamp"])

# Ensure sorted by time
df = df.sort_values("Timestamp").reset_index(drop=True)

TARGET_COL = "System_Demand"

df["hour"] = df["Timestamp"].dt.hour
df["dayofweek"] = df["Timestamp"].dt.dayofweek

# Feature set (same as LSTM script)
feature_cols = [
    "System_Demand",
    "USEP_Price_MWh",
    "PSI_North",
    "PSI_South",
    "PSI_East",
    "PSI_West",
    "PSI_Central",
    "hour",
    "dayofweek",
]

# Drop rows with missing features, just in case
df = df.dropna(subset=feature_cols).reset_index(drop=True)

print("Data loaded:", df.shape)
print(df.head())

# =============================================================
# 2. TRAIN / TEST SPLIT
# =============================================================

# Train: everything before 2019
train_df = df[df["Timestamp"] < "2019-01-01"]
# Test: year 2019
test_df  = df[(df["Timestamp"] >= "2019-01-01") & (df["Timestamp"] < "2020-01-01")]

# Scale ALL features together
scaler_all = MinMaxScaler()
train_scaled_all = scaler_all.fit_transform(train_df[feature_cols])
test_scaled_all  = scaler_all.transform(test_df[feature_cols])

# Separate scaler for the target so we can inverse-transform predictions
scaler_y = MinMaxScaler()
scaler_y.fit(train_df[[TARGET_COL]])

SEQ_LEN = 96  # 2 days = 96 half-hour points
target_index = feature_cols.index(TARGET_COL)  # 0, but explicit


def make_sequences(data, seq_len=96, target_col_index=0):
    """
    data: 2D array (n_samples, n_features) – scaled
    """
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i, :])         # past seq_len rows
        y.append(data[i, target_col_index])      # target at time i
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(train_scaled_all, SEQ_LEN, target_index)
X_test,  y_test  = make_sequences(test_scaled_all,  SEQ_LEN, target_index)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test.reshape(-1, 1),  dtype=torch.float32)

n_features = X_train.shape[2]


# =============================================================
# 3. TTM MODEL
# =============================================================

class TinyTimeMixer(nn.Module):
    def __init__(self, seq_len=96, n_features=1, hidden1=256, hidden2=128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(seq_len * n_features, hidden1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


model = TinyTimeMixer(seq_len=SEQ_LEN, n_features=n_features,
                      hidden1=256, hidden2=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

EPOCHS = 60         # a bit more training
BATCH_SIZE = 64
LR = 1e-3

dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = TinyTimeMixer(seq_len=SEQ_LEN, n_features=n_features,
                      hidden1=256, hidden2=128)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print("\nTraining TinyTimeMixer...")
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train MSE: {epoch_loss:.6f}")

# =============================================================
# 4. TEST FORECAST VISUALISATION (YEAR 2019)
# =============================================================

model.eval()
with torch.no_grad():
    pred_scaled = model(X_test_t).numpy()

pred = scaler_y.inverse_transform(pred_scaled)
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
plot_index = test_df["Timestamp"][SEQ_LEN:].reset_index(drop=True)

# Accuracy metrics
rmse = np.sqrt(mean_squared_error(y_true, pred))
mae  = mean_absolute_error(y_true, pred)
mape = np.mean(np.abs((y_true - pred) / y_true)) * 100
r2   = r2_score(y_true, pred)

def add_accuracy_box():
    text = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%\nR²: {r2:.3f}"
    plt.gca().text(
        0.02, 0.98, text, transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

# ================= TEST PLOTS (2019) =================

# 1) Full 2019
plt.figure(figsize=(16,6))
plt.plot(plot_index, y_true, label="Actual", linewidth=1)
plt.plot(plot_index, pred, label="TTM Forecast", linewidth=1.6)
plt.title("TinyTimeMixer Forecast — Year 2019")
plt.legend()
plt.xlabel("Date")
plt.ylabel("System Demand (MW)")
add_accuracy_box()
plt.tight_layout()
plt.show()

# 2) Last 7 days
plt.figure(figsize=(16,6))
plt.plot(plot_index[-48*7:], y_true[-48*7:], label="Actual", linewidth=1)
plt.plot(plot_index[-48*7:], pred[-48*7:], label="TTM Forecast", linewidth=1.6)
plt.title("TTM Forecast — Last 7 Days of 2019")
plt.legend()
plt.xlabel("Date")
plt.ylabel("System Demand (MW)")
add_accuracy_box()
plt.tight_layout()
plt.show()

# 3) Last 24 hours
plt.figure(figsize=(16,6))
plt.plot(plot_index[-48:], y_true[-48:], label="Actual", linewidth=1)
plt.plot(plot_index[-48:], pred[-48:], label="TTM Forecast", linewidth=1.6)
plt.title("TTM Forecast — Last 24 Hours of 2019")
plt.legend()
plt.xlabel("Date")
plt.ylabel("System Demand (MW)")
add_accuracy_box()
plt.tight_layout()
plt.show()

# =============================================================
# 5. FULL YEAR 2019 PREDICTION
# =============================================================

full_2019 = df[(df["Timestamp"] >= "2019-01-01") & (df["Timestamp"] < "2020-01-01")]
full_scaled_all = scaler_all.transform(full_2019[feature_cols])

X_full, y_full = make_sequences(full_scaled_all, SEQ_LEN, target_index)
X_full_t = torch.tensor(X_full, dtype=torch.float32)

with torch.no_grad():
    full_pred_scaled = model(X_full_t).numpy()

full_pred = scaler_y.inverse_transform(full_pred_scaled)              # (N, 1)
full_true = scaler_y.inverse_transform(y_full.reshape(-1, 1))         # (N, 1)
full_index = full_2019["Timestamp"][SEQ_LEN:].reset_index(drop=True)

# =============================================================
# 6. MONTHLY AVERAGE (CLEAN JAN–DEC GRAPH)
# =============================================================

monthly_df = pd.DataFrame({
    "Timestamp": full_index,
    "Actual": full_true.flatten(),
    "Forecast": full_pred.flatten()
})

monthly_df["Month"] = monthly_df["Timestamp"].dt.month
monthly_avg = monthly_df.groupby("Month").mean(numeric_only=True)

# Clean Jan–Dec monthly average plot
plt.figure(figsize=(12,5))
plt.plot(
    monthly_avg.index, 
    monthly_avg["Actual"], 
    marker="o", 
    linewidth=2, 
    label="Actual (Monthly Avg)",
)
plt.plot(monthly_avg.index, monthly_avg["Forecast"], marker="o", linewidth=2, label="TTM Forecast (Monthly Avg)")

plt.title("TinyTimeMixer Forecast — Monthly Average (Jan–Dec 2019)")
plt.xlabel("Month (1–12)")
plt.ylabel("System Demand (MW)")
plt.xticks(range(1, 13))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("\nMonthly Averages Plot Completed Successfully!")