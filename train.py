# train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
import pickle
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings ("ignore")  # Suppress minor warnings

# Load environment variables
load_dotenv()

# Configuration from .env
DATA_PATH = os.getenv("DATA_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")
PREDICTION_HORIZON = int(os.getenv("PREDICTION_HORIZON"))
PRICE_CHANGE_THRESHOLD = float(os.getenv("PRICE_CHANGE_THRESHOLD"))
TRAIN_SPLIT_RATIO = float(os.getenv("TRAIN_SPLIT_RATIO"))
FEATURES = os.getenv("FEATURES").split(",")
MOVING_AVERAGE_WINDOWS = [int(x) for x in os.getenv("MOVING_AVERAGE_WINDOWS").split(",")]
RSI_WINDOW = int(os.getenv("RSI_WINDOW"))
LAG_PERIODS = [int(x) for x in os.getenv("LAG_PERIODS").split(",")]
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
CLASS_WEIGHT = os.getenv("CLASS_WEIGHT")

# Step 1: Load data
print("Loading data...")
df = pd.read_pickle(DATA_PATH)

print("First few rows:")
print(df.head())

# Step 2: Preprocess data
print("Preprocessing data...")
df.index = pd.to_datetime(df.index)
df = df.drop(['volume_abs_gex', 'volume_net_gex'], axis=1)

# Create target variable
df['future_close'] = df['Spot_Close'].shift(-PREDICTION_HORIZON)
df['price_change'] = df['future_close'] - df['Spot_Close']
df['signal'] = 0
df.loc[df['price_change'] > PRICE_CHANGE_THRESHOLD, 'signal'] = 1   # Buy
df.loc[df['price_change'] < -PRICE_CHANGE_THRESHOLD, 'signal'] = -1  # Sell
df = df.dropna()

# Step 3: Engineer features
print("Engineering features...")
# Price-based features
df['price_diff'] = df['Spot_Close'] - df['Spot_Open']
df['price_range'] = df['Spot_High'] - df['Spot_Low']
for window in MOVING_AVERAGE_WINDOWS:
    df[f'ma{window}'] = df['Spot_Close'].rolling(window=window).mean()
df['rsi'] = RSIIndicator(df['Spot_Close'], window=RSI_WINDOW).rsi()
# GEX and EPS features
df['net_gex_diff'] = df['open_net_gex'].diff()
df['abs_gex_diff'] = df['open_abs_gex'].diff()
df['eps_diff'] = df['PCT_EPS_1mo_Close'].diff()
# Lagged features
for lag in LAG_PERIODS:
    df[f'lag{lag}_close'] = df['Spot_Close'].shift(lag)
df = df.dropna()

# Step 4: Prepare training data
X = df[FEATURES]
y = df['signal']
train_size = int(TRAIN_SPLIT_RATIO * len(df))
X_train = X[:train_size]
y_train = y[:train_size]

# Step 5: Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled, columns=FEATURES, index=X_train.index)

# Step 6: Train model
print("Training model...")
model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    class_weight=CLASS_WEIGHT
)

model.fit(X_train, y_train)

# Step 7: Save model and scaler
print("Saving model and scaler...")
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print("Training completed!")