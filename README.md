# StockSignal

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Issues](https://img.shields.io/github/issues/makqai/StockSignal)

**StockSignal** is a Machine Learning project that generates **Buy**, **Sell**, and **Hold** signals for Apple (AAPL) stock using minute-by-minute trading data. It leverages features like price movements, options-related metrics (Gamma Exposure, Expected Price Swing), and technical indicator (RSI) to predict short-term price changes. The project is split into three stages—training, evaluation, and inference—making it modular and easy to extend.

Perfect for those exploring algorithmic trading or ML, this repo provides a practical example of applying Random Forest to stock data, with a focus on intraday predictions.

## Table of Contents

- [StockSignal](#stocksignal)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Configuration](#configuration)
  - [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Inference](#inference)
  - [Sample Output](#sample-output)
    - [Evaluation](#evaluation-1)
    - [Inference](#inference-1)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
    - [Happy trading and coding! 🚀](#happy-trading-and-coding-)

## Features

- **ML-based Trading Signals**: Predicts Buy/Sell/Hold signals for AAPL, MSFT, and SPY using a Random Forest model.
- **Minute-by-Minute Data**: Analyzes high-frequency data with price, GEX, EPS, and technical indicators.
- **Modular Workflow**:
  - `train.py`: Preprocesses data, engineers features, and trains the model.
  - `evaluate.py`: Assesses model performance with metrics like precision and recall.
  - `infer.py`: Generates signals, backtests trading strategy, and visualizes results.
- **Configurable**: All settings (e.g., prediction horizon, features) are managed via a `.env` file.
- **Backtesting**: Simulates trading to estimate returns, including transaction costs.
- **Extensible**: Easily adapt for other stocks or models (e.g., XGBoost, LSTM).

## Getting Started

Follow these steps to set up and run StockSignal on your machine.

### Prerequisites

- **Python 3.12+**: Install from [python.org](https://www.python.org/downloads/).
- **Git**: To clone the repository.
- A Pickle file with minute-by-minute stock data (e.g., `NetGEX_AbsGEX_EPS(XXXX).pickle`).

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/makqai/StockSignal.git
   cd StockSignal
   ```

2. **Install dependencies**:
   
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Set up environment variables**:

- Copy the example .env file:

  ```bash
  cp .env.example .env
  ```

- Edit .env to specify your data path and other settings:

  ```bash
  DATA_PATH=your_data.pickle
  MODEL_PATH=model.pkl
  SCALER_PATH=scaler.pkl
  PREDICTION_HORIZON=5
  PRICE_CHANGE_THRESHOLD=0.20
  ```

- Refer to .env.example for all configurable options (e.g., features, model parameters).

## Usage

The project is divided into three scripts, run in sequence:

### Training

Train the Random Forest model and save it:

  ```bash
  python train.py
  ```

### Evaluation

Evaluate the model’s performance:

  ```bash
  python evaluate.py
  ```

### Inference

Generate signals and backtest:

  ```bash
  python infer.py
  ```

## Sample Output

### Evaluation

**Classification Report**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Sell  | 0.05      | 0.00   | 0.00     | 1186    |
| Hold  | 0.93      | 1.00   | 0.96     | 31093   |
| Buy   | 0.00      | 0.00   | 0.00     | 1187    |

**Accuracy**: 0.93  
**Macro Avg F1-Score**: 0.32  
**Weighted Avg F1-Score**: 0.89

**Confusion Matrix**

|       | Sell | Hold  | Buy  |
|-------|------|-------|------|
| Sell  | 3    | 1183  | 0    |
| Hold  | 54   | 31037 | 2    |
| Buy   | 2    | 1185  | 0    |

**Feature Importance**

- `price_range`: 0.144925
- `open_abs_gex`: 0.079646
- `rsi`: 0.070754
- ...

### Inference

**Backtest Results**

- **Initial Capital**: $10,000.00
- **Final Capital**: $10,028.57
- **Returns**: 0.29%
- **Number of Trades**: 4


## License
This project is licensed under the MIT.

## Acknowledgments
- Inspired by algorithmic trading and ML communities.
- Thanks to libraries like scikit-learn, pandas, and ta.
- Built with guidance from open-source best practices.


### Happy trading and coding! 🚀