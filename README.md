# Cryptocurrency Trading Bot with Transformer Model

A production-grade Python trading bot that predicts BTC/USDT closing prices using a PyTorch Transformer model.

## Project Structure

The codebase is organized into modular components:

```
transformer-agent/
├── crypto_trading_bot.py    # Main entry point
├── config.py                 # Configuration constants
├── transformer_model.py      # Transformer model architecture
├── data_fetcher.py           # Data fetching from Binance
├── feature_engineering.py    # Technical indicators & features
├── dataset.py                # Custom PyTorch Dataset
├── trainer.py                # Training utilities
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Module Descriptions

### `config.py`
Centralized configuration file containing:
- Trading parameters (symbol, timeframe, limits)
- Model hyperparameters
- Training settings
- Device configuration (CPU/GPU)

### `transformer_model.py`
Implements the Transformer architecture:
- Input embedding layer
- Learnable positional encoding
- Multi-layer Transformer encoder
- Global average pooling
- Decoder head with dropout

### `data_fetcher.py`
Handles all data fetching operations:
- `fetch_massive_history()`: Paginated fetching of 50k+ historical candles
- `fetch_latest_update()`: Efficient fetching of recent candles for live predictions

### `feature_engineering.py`
Feature calculation and preprocessing:
- Technical indicators (RSI, MACD, ATR)
- Log returns and price ratios
- Volume features
- Sequence preparation for training

### `dataset.py`
Custom PyTorch Dataset class for price sequences and targets.

### `trainer.py`
Training utilities:
- Model training loop with validation
- Loss tracking and progress reporting

### `crypto_trading_bot.py`
Main orchestration script that:
1. Fetches historical data
2. Trains the model
3. Runs live prediction loop

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python crypto_trading_bot.py
```

The script will:
1. Fetch 50,000 historical candles (takes a few minutes)
2. Train the Transformer model (~20 epochs)
3. Enter live prediction mode with 15-minute updates

## Configuration

Edit `config.py` to customize:
- Trading pair and timeframe
- Model architecture parameters
- Training hyperparameters
- Binance API credentials (optional for public endpoints)

## Features

- ✅ Paginated data fetching (handles 50k+ candles)
- ✅ Technical indicators (RSI, MACD, ATR)
- ✅ RobustScaler for feature normalization
- ✅ Transformer model with learnable positional encoding
- ✅ Device-agnostic (auto-detects CUDA/CPU)
- ✅ Live prediction loop with signal generation
- ✅ Error handling and retry logic

## Signal Generation

- **STRONG BUY**: Predicted price > current price + 0.1%
- **STRONG SELL**: Predicted price < current price - 0.1%
- **HOLD**: Otherwise

## Notes

- No API keys required for data fetching (uses public Binance endpoints)
- Model trains on historical data once at startup
- Live predictions run every 15 minutes
- Press Ctrl+C to stop the bot gracefully

