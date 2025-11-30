"""
Feature engineering functions for technical indicators and feature calculation
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Tuple, List
from config import USE_EXTENDED_FEATURES, USE_FUNDAMENTAL_DATA


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators and features
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Technical indicators using pandas_ta
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    if macd is not None and not macd.empty:
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDS_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
    else:
        df['macd'] = 0.0
        df['macd_signal'] = 0.0
        df['macd_hist'] = 0.0
    
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Additional simple technical indicators (always calculated, fast)
    # ADX - Trend strength
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx is not None and not adx.empty:
        df['adx'] = adx['ADX_14']
        df['adx_pos'] = adx['DMP_14']  # +DI
        df['adx_neg'] = adx['DMN_14']  # -DI
    else:
        df['adx'] = 25.0
        df['adx_pos'] = 25.0
        df['adx_neg'] = 25.0
    
    # OBV - On-Balance Volume
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['obv_ma'] = df['obv'].rolling(window=20).mean()
    df['obv_ratio'] = df['obv'] / (df['obv_ma'] + 1e-8)
    
    # Extended features (optional - can be disabled for faster training)
    if USE_EXTENDED_FEATURES:
        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None and not bbands.empty:
            df['bb_upper'] = bbands['BBU_20_2.0']
            df['bb_middle'] = bbands['BBM_20_2.0']
            df['bb_lower'] = bbands['BBL_20_2.0']
            df['bb_width'] = (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / bbands['BBM_20_2.0']
            df['bb_position'] = (df['close'] - bbands['BBL_20_2.0']) / (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0'] + 1e-8)
        else:
            df['bb_upper'] = df['close']
            df['bb_middle'] = df['close']
            df['bb_lower'] = df['close']
            df['bb_width'] = 0.0
            df['bb_position'] = 0.5
        
        # Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None and not stoch.empty:
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
        else:
            df['stoch_k'] = 50.0
            df['stoch_d'] = 50.0
        
        # EMA Crossovers
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)
        df['ema_cross'] = (df['ema_12'] - df['ema_26']) / (df['ema_26'] + 1e-8)
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Price features (normalized)
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Volume features
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    
    # Fundamental data (optional - requires API call, slower)
    if USE_FUNDAMENTAL_DATA:
        try:
            from fundamental_data import add_fear_greed_to_df
            df = add_fear_greed_to_df(df, use_current=True)
        except Exception as e:
            print(f"Warning: Could not add Fear & Greed Index: {e}")
            df['fear_greed'] = 50.0
            df['fear_greed_norm'] = 0.5
    
    # Drop NaN values
    df = df.dropna()
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get the list of feature columns to use for training
    
    Args:
        df: DataFrame with calculated features
        
    Returns:
        List of feature column names
    """
    # Base features (always included)
    feature_columns = [
        # Price data
        'open', 'high', 'low', 'close', 'volume',
        # Technical indicators
        'rsi', 'macd', 'macd_signal', 'macd_hist', 'atr',
        # ADX (trend strength)
        'adx', 'adx_pos', 'adx_neg',
        # OBV (volume momentum)
        'obv_ratio',
        # Price features
        'log_return', 'high_low_ratio', 'close_open_ratio',
        # Volume features
        'volume_ratio'
    ]
    
    # Extended features (optional)
    if USE_EXTENDED_FEATURES:
        extended_features = [
            # Bollinger Bands
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            # Stochastic
            'stoch_k', 'stoch_d',
            # EMA
            'ema_12', 'ema_26', 'ema_cross',
            # Time features (cyclical)
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        feature_columns.extend(extended_features)
    
    # Fundamental features (optional)
    if USE_FUNDAMENTAL_DATA:
        feature_columns.append('fear_greed_norm')
    
    # Filter to only columns that exist
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    return feature_columns


def prepare_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    feature_columns: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences and targets for training
    
    Args:
        df: DataFrame with features
        sequence_length: Length of input sequences
        feature_columns: List of feature column names
        
    Returns:
        Tuple of (sequences, targets) as numpy arrays
    """
    sequences = []
    targets = []
    
    for i in range(sequence_length, len(df)):
        # Input sequence
        seq = df[feature_columns].iloc[i-sequence_length:i].values
        sequences.append(seq)
        
        # Target: next closing price
        target = df['close'].iloc[i]
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

