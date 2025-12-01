# Adapted from:
# "Multivariate Time Series Forecasting with LSTMs in Keras" (GeeksforGeeks)
# Customised for Singapore power demand data
# Target: System Demand (Actual)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# ======================================================
# 1. LOAD DATA
# ======================================================

# This file lives in: project/LSTM/LSTM-GFG.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "datasets"

file_path = DATA_DIR / "forecast_dataset.csv"   # <--- uses forecast file now
print(f"[INFO] Loading data from: {file_path}")

df = pd.read_csv(file_path, parse_dates=["Timestamp"])

# ======================================================
# 2. BASIC PREPROCESSING (timestamp + numeric columns)
# ======================================================

# Sort by time just in case
df = df.sort_values('Timestamp').reset_index(drop=True)

print("\n[OK] Dtypes after cleaning:")
print(df.dtypes)

# ======================================================
# 3. SELECT FEATURES & TARGET (MULTIVARIATE)
#    - Use past values of ALL variables to predict
#      next System Demand (Actual)
# ======================================================

target_col = "System_Demand"

feature_cols = [
    "System_Demand",     # include target history as an input
    "USEP_Price_MWh",
    "PSI_North",
    "PSI_South",
    "PSI_East",
    "PSI_West",
    "PSI_Central",
]

data = df[feature_cols].values
print("\n[OK] Feature matrix shape:", data.shape)

n_features = data.shape[1]       # now 7 features
target_index = feature_cols.index(target_col)  # 0, but clearer & safer

# ======================================================
# 4. SCALE DATA (MIN-MAX) – like in GFG example
# ======================================================

scaler_all = MinMaxScaler()
data_scaled = scaler_all.fit_transform(data)

# Separate scaler for target so we can invert predictions later
scaler_y = MinMaxScaler()
scaler_y.fit(data[:, target_index].reshape(-1, 1))

print("[OK] Scaled data shape:", data_scaled.shape)

# ======================================================
# 5. CONVERT TO SUPERVISED FORM (SEQUENCE CREATION)
#    Similar to GFG create_dataset / sampler function
# ======================================================

def create_sequences(dataset, time_steps=48, target_col_index=0):
    """
    dataset: 2D array of shape (n_samples, n_features) – scaled
    time_steps: how many past time steps to look back
    target_col_index: which column to use as prediction target
    """
    X, y = [], []
    for i in range(time_steps, len(dataset)):
        # past 'time_steps' rows as input
        X.append(dataset[i - time_steps:i, :])
        # value at time i for the target column
        y.append(dataset[i, target_col_index])
    return np.array(X), np.array(y)

TIME_STEPS = 48   # 48 half-hours = 24 hours history

X_all, y_all = create_sequences(data_scaled, TIME_STEPS, target_index)
print("[OK] Sequence shapes -> X:", X_all.shape, "| y:", y_all.shape)

# ======================================================
# 6. TRAIN / TEST SPLIT (e.g. 80% / 20%) – like tutorial
# ======================================================

train_size = int(len(X_all) * 0.8)

X_train, X_test = X_all[:train_size], X_all[train_size:]
y_train, y_test = y_all[:train_size], y_all[train_size:]

# Keras expects y to be 2D (samples, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print("[OK] Train shapes:", X_train.shape, y_train.shape)
print("[OK] Test shapes :", X_test.shape, y_test.shape)

# ======================================================
# 7. BUILD LSTM MODEL (Keras Sequential) – GFG-style
# ======================================================

model = Sequential()
model.add(LSTM(
    units=64,                 # GFG often uses a single big LSTM; 64 is a
                              # lighter custom choice for your dataset
    return_sequences=True,
    input_shape=(TIME_STEPS, n_features)
))
model.add(Dropout(0.2))

model.add(LSTM(units=32))
model.add(Dropout(0.2))

model.add(Dense(1))           # single target: System Demand (Actual)

model.compile(optimizer='adam', loss='mse')
model.summary()

# ======================================================
# 8. TRAIN LSTM
# ======================================================

history = model.fit(
    X_train, y_train,
    epochs=10,                # similar to tutorial; adjust if needed
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ======================================================
# 9. MAKE PREDICTIONS & INVERSE SCALE
# ======================================================

y_pred_scaled = model.predict(X_test)

# Inverse transform from scaled [0,1] back to MW
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_real = scaler_y.inverse_transform(y_test)

print("\n[OK] Prediction shapes:", y_pred.shape, y_test_real.shape)

# ======================================================
# 10. PLOT – ACTUAL vs PREDICTED (FIRST 500 POINTS)
# ======================================================

plt.figure(figsize=(12, 6))
plt.plot(y_test_real[:500], label='Actual Demand')
plt.plot(y_pred[:500], label='Predicted Demand')
plt.title('LSTM Forecast vs Actual (First 500 Test Points)')
plt.xlabel('Time Step (half-hour slots in test set)')
plt.ylabel('System Demand (MW)')
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

# --- MAPE ---
mape = mean_absolute_percentage_error(y_test_real, y_pred) * 100

# --- Accuracy ---
accuracy = 100 - mape

# --- MAE ---
mae = mean_absolute_error(y_test_real, y_pred)

# --- RMSE ---
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))

print("====== LSTM Forecast Performance ======")
print(f"MAPE:      {mape:.4f}%")
print(f"Accuracy:  {accuracy:.4f}%")
print(f"MAE:       {mae:.4f} MW")
print(f"RMSE:      {rmse:.4f} MW")

from sklearn.metrics import r2_score

r2 = r2_score(y_test_real, y_pred)
print("R² Score:", r2)


