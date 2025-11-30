"""
Fundamental data fetching from free APIs
"""

import requests
import pandas as pd
import time
from typing import Optional
from datetime import datetime


def fetch_fear_greed_index() -> Optional[int]:
    """
    Fetch current Fear & Greed Index from Alternative.me API (FREE)
    
    Returns:
        Fear & Greed Index value (0-100) or None if fetch fails
        0-24: Extreme Fear
        25-49: Fear
        50-74: Greed
        75-100: Extreme Greed
    """
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and len(data['data']) > 0:
            value = int(data['data'][0]['value'])
            return value
        return None
    except Exception as e:
        print(f"Warning: Could not fetch Fear & Greed Index: {e}")
        return None


def add_fear_greed_to_df(df: pd.DataFrame, use_current: bool = True) -> pd.DataFrame:
    """
    Add Fear & Greed Index to dataframe
    
    Args:
        df: DataFrame with datetime index
        use_current: If True, use current F&G value for all rows (for historical data)
                    If False, would fetch historical values (not available in free API)
        
    Returns:
        DataFrame with 'fear_greed' column added
    """
    df = df.copy()
    
    if use_current:
        # For historical training data, use current F&G value
        # (Free API doesn't provide historical, so we use current as proxy)
        current_fng = fetch_fear_greed_index()
        
        if current_fng is not None:
            # Add to all rows (since it's a daily indicator)
            df['fear_greed'] = float(current_fng)
            df['fear_greed_norm'] = df['fear_greed'] / 100.0
        else:
            # Default to neutral (50) if fetch fails
            df['fear_greed'] = 50.0
            df['fear_greed_norm'] = 0.5
    else:
        # For live predictions, fetch fresh value
        current_fng = fetch_fear_greed_index()
        if current_fng is not None:
            df['fear_greed'] = float(current_fng)
            df['fear_greed_norm'] = df['fear_greed'] / 100.0
        else:
            df['fear_greed'] = 50.0
            df['fear_greed_norm'] = 0.5
    
    return df


def fetch_dxy_index() -> Optional[float]:
    """
    Fetch Dollar Index (DXY) from FRED API (FREE, but needs API key)
    For now, we'll use a simpler approach with yfinance
    
    Returns:
        DXY value or None if fetch fails
    """
    try:
        import yfinance as yf
        # Get DXY ticker
        dxy = yf.Ticker("DX-Y.NYB")
        hist = dxy.history(period="1d")
        
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        return None
    except Exception as e:
        print(f"Warning: Could not fetch DXY: {e}")
        return None


def add_dxy_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Dollar Index (DXY) to dataframe
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with 'dxy' column added
    """
    df = df.copy()
    
    current_dxy = fetch_dxy_index()
    
    if current_dxy is not None:
        df['dxy'] = float(current_dxy)
        # Normalize (DXY typically ranges 90-110)
        df['dxy_norm'] = (df['dxy'] - 90) / 20.0  # Normalize to roughly 0-1
        df['dxy'] = df['dxy'].ffill()
        df['dxy_norm'] = df['dxy_norm'].ffill()
    else:
        # Default value if fetch fails
        df['dxy'] = 100.0
        df['dxy_norm'] = 0.5
    
    return df

