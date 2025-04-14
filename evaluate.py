# evaluate.py
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from ta.momentum import RSIIndicator
import pickle
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

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

# Step 1: Load data
print("Loading data...")
df = pd.read_pickle(DATA_PATH)

# Step 2: Preprocess data
print("Preprocessing data...")
df.index = pd.to_datetime(df.index)
df = df.drop(['volume_abs_gex', 'volume_net_gex'], axis=1)
df['future_close'] = df['Spot_Close'].shift(-PREDICTION_HORIZON)
df['price_change'] = df['future_close'] - df['Spot_Close']
df['signal'] = 0
df.loc[df['price_change'] > PRICE_CHANGE_THRESHOLD, 'signal'] = 1
df.loc[df['price_change'] < -PRICE_CHANGE_THRESHOLD, 'signal'] = -1
df = df.dropna()

# Step 3: Engineer features
print("Engineering features...")
df['price_diff'] = df['Spot_Close'] - df['Spot_Open']
df['price_range'] = df['Spot_High'] - df['Spot_Low']
for window in MOVING_AVERAGE_WINDOWS:
    df[f'ma{window}'] = df['Spot_Close'].rolling(window=window).mean()
df['rsi'] = RSIIndicator(df['Spot_Close'], window=RSI_WINDOW).rsi()
df['net_gex_diff'] = df['open_net_gex'].diff()
df['abs_gex_diff'] = df['open_abs_gex'].diff()
df['eps_diff'] = df['PCT_EPS_1mo_Close'].diff()
for lag in LAG_PERIODS:
    df[f'lag{lag}_close'] = df['Spot_Close'].shift(lag)
df = df.dropna()

# Step 4: Prepare test data
X = df[FEATURES]
y = df['signal']
train_size = int(TRAIN_SPLIT_RATIO * len(df))
X_test = X[train_size:]
y_test = y[train_size:]

# Step 5: Load model and scaler
print("Loading model and scaler...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Step 6: Scale test data
print("Scaling test data...")
X_test_scaled = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_scaled, columns=FEATURES, index=X_test.index)

# Step 7: Evaluate model
print("Evaluating model...")
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Sell', 'Hold', 'Buy']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_importance)

print("Evaluation completed!")