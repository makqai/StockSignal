# infer.py
import pandas as pd
import numpy as np
import pickle
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
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
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL"))
TRANSACTION_COST = float(os.getenv("TRANSACTION_COST"))

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

# Step 4: Prepare inference data (test set)
train_size = int(TRAIN_SPLIT_RATIO * len(df))
test_df = df[train_size:].copy()
X_test = test_df[FEATURES]

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

# Step 7: Generate signals
print("Generating signals...")
test_df['predicted_signal'] = model.predict(X_test)

# Step 8: Backtest the strategy
print("Running backtest...")
capital = INITIAL_CAPITAL
position = 0  # Number of shares held
trades = []

for i in test_df.index:
    signal = test_df.loc[i, 'predicted_signal']
    price = test_df.loc[i, 'Spot_Close']
    
    if signal == 1 and position == 0:  # Buy
        shares = capital // price
        if shares > 0:
            position = shares
            capital -= shares * price
            capital -= shares * TRANSACTION_COST
            trades.append(('Buy', price, i))
    
    elif signal == -1 and position > 0:  # Sell
        capital += position * price
        capital -= position * TRANSACTION_COST
        trades.append(('Sell', price, i))
        position = 0

# Close any open position
if position > 0:
    capital += position * test_df['Spot_Close'].iloc[-1]
    capital -= position * TRANSACTION_COST
    trades.append(('Sell', test_df['Spot_Close'].iloc[-1], test_df.index[-1]))
    position = 0

final_capital = capital
returns = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
print(f"\nBacktest Results:")
print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
print(f"Final Capital: ${final_capital:.2f}")
print(f"Returns: {returns:.2f}%")
print(f"Number of trades: {len(trades)}")

# Step 9: Visualize results
print("Generating plot...")
plt.figure(figsize=(14, 7))
plt.plot(test_df['timestamp'], test_df['Spot_Close'], label='Spot Close', color='blue')
buy_signals = test_df[test_df['predicted_signal'] == 1]
plt.scatter(buy_signals['timestamp'], buy_signals['Spot_Close'], 
           label='Buy Signal', color='green', marker='^', s=100)
sell_signals = test_df[test_df['predicted_signal'] == -1]
plt.scatter(sell_signals['timestamp'], sell_signals['Spot_Close'], 
           label='Sell Signal', color='red', marker='v', s=100)
plt.title('AAPL Stock Price with Buy/Sell Signals')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Inference completed!")