# StockSignalML

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Issues](https://img.shields.io/github/issues/makqai/StockSignal)

**StockSignal** is a Machine Learning project that generates **Buy**, **Sell**, and **Hold** signals for Apple (AAPL) stock using minute-by-minute trading data. It leverages features like price movements, options-related metrics (Gamma Exposure, Expected Price Swing), and technical indicator (RSI) to predict short-term price changes. The project is split into three stages—training, evaluation, and inference—making it modular and easy to extend.

Perfect for those exploring algorithmic trading or ML, this repo provides a practical example of applying Random Forest to stock data, with a focus on intraday predictions.

## Table of Contents

- [StockSignalML](#stocksignalml)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)

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

- **Python 3.7+**: Install from [python.org](https://www.python.org/downloads/).
- **Git**: To clone the repository.
- A Pickle file with minute-by-minute stock data (e.g., `NetGEX_AbsGEX_EPS(XXXX).pickle`).

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/makqai/StockSignal.git
   cd StockSignal