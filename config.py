"""
Configuration constants for the cryptocurrency trading bot
"""

import torch

# Trading Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '5m'  # Changed from 15m to 5m for faster trading signals
TRAINING_LIMIT = 50000  # Number of candles for training
LIVE_UPDATE_LIMIT = 100  # Number of candles for live updates

# Model Configuration
SEQUENCE_LENGTH = 60  # Number of timesteps to use for prediction
BATCH_SIZE = 64  # Increase to 128-256 for GPU training (more VRAM available)
EPOCHS = 20  # Can increase to 30-50 for GPU training (faster, can train longer)
LEARNING_RATE = 0.001

# Model Architecture
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Binance API Configuration (optional - leave empty for public endpoints)
BINANCE_API_KEY = ''
BINANCE_SECRET = ''

# Feature Configuration
USE_EXTENDED_FEATURES = True  # Set to True to use Bollinger Bands, Stochastic, EMA, etc. (28 features total)
USE_FUNDAMENTAL_DATA = True  # Set to True to use Fear & Greed Index (30 features total, slower, needs API)

