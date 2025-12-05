"""
Data fetching functions for cryptocurrency market data
"""

import time
import os
from typing import Optional
from datetime import datetime
import ccxt
import pandas as pd
from config import TIMEFRAME, LIVE_UPDATE_LIMIT


def fetch_massive_history(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int
) -> pd.DataFrame:
    """
    Fetch historical data with pagination to get large amounts of data
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair symbol
        timeframe: Timeframe string
        limit: Total number of candles to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching {limit} candles of historical data...")
    
    all_candles = []
    batch_size = 1000  # Binance max per request
    
    # Calculate timeframe duration in milliseconds
    timeframe_ms = {
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000
    }.get(timeframe, 5 * 60 * 1000)
    
    # Start from current time and work backwards
    since = None
    
    while len(all_candles) < limit:
        try:
            # Calculate how many candles we still need
            remaining = limit - len(all_candles)
            current_batch_size = min(batch_size, remaining)
            
            # Fetch batch
            if since is None:
                # First request: get most recent data
                candles = exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    limit=current_batch_size
                )
            else:
                # Subsequent requests: fetch older data
                candles = exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=since,
                    limit=current_batch_size
                )
            
            if not candles or len(candles) == 0:
                print("No more data available")
                break
            
            # If this is the first batch, prepend it
            # Otherwise, prepend to get chronological order (oldest first)
            if since is None:
                all_candles = candles + all_candles
            else:
                all_candles = candles + all_candles
            
            # Update 'since' to go further back in time
            # Use the oldest candle timestamp minus one interval
            oldest_timestamp = candles[0][0]
            since = oldest_timestamp - timeframe_ms
            
            print(f"Fetched {len(all_candles)}/{limit} candles...")
            
            # Rate limiting
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5)
            continue
    
    # Limit to requested amount and sort by timestamp
    all_candles = sorted(all_candles[:limit], key=lambda x: x[0])
    
    # Convert to DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime').sort_index()
    
    # Automatically save to files
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate filename with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_clean = symbol.replace('/', '_')
    filename_base = f"data/{symbol_clean}_{timeframe}_{timestamp_str}"
    
    # Save as CSV (easy to read)
    csv_file = f"{filename_base}.csv"
    df.to_csv(csv_file)
    print(f"Data saved to CSV: {csv_file}")
    
    # Save as JSON (for API/other uses)
    json_file = f"{filename_base}.json"
    # Reset index to include datetime in JSON
    df_json = df.reset_index()
    df_json['datetime'] = df_json['datetime'].astype(str)  # Convert datetime to string
    df_json.to_json(json_file, orient='records', indent=2)
    print(f"Data saved to JSON: {json_file}")
    
    print(f"Successfully fetched {len(df)} candles")
    return df


def fetch_latest_update(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int = LIVE_UPDATE_LIMIT
) -> pd.DataFrame:
    """
    Fetch the latest candles for live prediction
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair symbol
        timeframe: Timeframe string
        limit: Number of recent candles to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        df = pd.DataFrame(
            candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime').sort_index()
        
        return df
        
    except Exception as e:
        print(f"Error fetching latest data: {e}")
        raise

