"""
Run inference only - Use pre-trained model for live predictions
No training, just load saved model and predict
"""

import time
import warnings
from datetime import datetime

import ccxt
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
import pickle

# Import modular components
from config import (
    SYMBOL, TIMEFRAME, LIVE_UPDATE_LIMIT,
    SEQUENCE_LENGTH, DEVICE
)
from transformer_model import TransformerModel
from data_fetcher import fetch_latest_update
from feature_engineering import calculate_features, get_feature_columns
from dataset import PriceDataset

warnings.filterwarnings('ignore')

print(f"Using device: {DEVICE}")


def load_trained_model(model_path: str = 'trained_model.pth'):
    """
    Load trained model and scaler from saved file
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        model, scaler, feature_columns, model_config
    """
    print(f"Loading trained model from {model_path}...")
    
    try:
        # Allow loading of the RobustScaler object saved in the checkpoint.
        # This is safe here because the checkpoint is created by our own code.
        try:
            torch.serialization.add_safe_globals([RobustScaler])
        except Exception:
            # Older PyTorch versions may not have add_safe_globals; ignore in that case.
            pass

        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        # Extract components
        model_state = checkpoint['model_state_dict']
        scaler = checkpoint['scaler']
        feature_columns = checkpoint['feature_columns']
        model_config = checkpoint['config']
        
        # Initialize model with saved config
        model = TransformerModel(
            input_dim=model_config['input_dim'],
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout'],
            max_seq_length=model_config['max_seq_length']
        )
        
        # Load weights
        model.load_state_dict(model_state)
        model.to(DEVICE)
        model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Features: {len(feature_columns)}")
        print(f"Model config: {model_config}")
        
        return model, scaler, feature_columns, model_config
        
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first using: python crypto_trading_bot.py")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None


def main():
    """Main execution function for inference only"""
    
    # Load trained model
    model, scaler, feature_columns, model_config = load_trained_model()
    
    if model is None:
        return
    
    # Initialize exchange
    from config import BINANCE_API_KEY, BINANCE_SECRET
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot'
        }
    })
    
    # Calculate sleep interval based on timeframe
    timeframe_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440
    }.get(TIMEFRAME, 15)
    
    # Live prediction loop
    print("=" * 60)
    print("Starting live prediction loop (Inference Only)")
    print("=" * 60)
    print(f"Monitoring {SYMBOL} on {TIMEFRAME} timeframe")
    print(f"Sleep interval: {timeframe_minutes} minutes\n")
    
    while True:
        try:
            # Fetch latest data
            df_live = fetch_latest_update(
                exchange,
                SYMBOL,
                TIMEFRAME,
                LIVE_UPDATE_LIMIT
            )
            
            # Calculate features
            df_live_features = calculate_features(df_live)
            
            # Check if we have enough data
            if len(df_live_features) < SEQUENCE_LENGTH:
                print(f"Not enough data. Have {len(df_live_features)}, need {SEQUENCE_LENGTH}")
                time.sleep(timeframe_minutes * 60)
                continue
            
            # Verify all required features exist
            missing_features = [col for col in feature_columns if col not in df_live_features.columns]
            if missing_features:
                print(f"Warning: Missing features {missing_features}. Skipping this iteration.")
                time.sleep(timeframe_minutes * 60)
                continue
            
            # Prepare sequence (last SEQUENCE_LENGTH rows)
            latest_sequence = df_live_features[feature_columns].iloc[-SEQUENCE_LENGTH:].values
            
            # Scale using pre-fitted scaler
            latest_sequence_scaled = scaler.transform(
                latest_sequence.reshape(-1, len(feature_columns))
            ).reshape(1, SEQUENCE_LENGTH, len(feature_columns))
            
            # Convert to tensor and predict
            sequence_tensor = torch.FloatTensor(latest_sequence_scaled).to(DEVICE)
            
            with torch.no_grad():
                prediction = model(sequence_tensor).cpu().item()
            
            # Get current price
            current_price = df_live_features['close'].iloc[-1]
            
            # Model predicts percentage return, convert to price prediction
            # prediction is already in percentage (e.g., 0.5 means +0.5%)
            price_change_pct = prediction
            predicted_price = current_price * (1 + prediction / 100.0)
            
            # Generate signal
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Current Price: ${current_price:.2f}")
            print(f"Predicted Price: ${predicted_price:.2f}")
            print(f"Expected Change: {price_change_pct:.2f}%")
            
            # Scalping thresholds (tighter for 5m timeframe)
            if price_change_pct > 0.05:  # 0.05% for scalping
                print(">>> SIGNAL: STRONG BUY <<<")
            elif price_change_pct < -0.05:  # -0.05% for scalping
                print(">>> SIGNAL: STRONG SELL <<<")
            else:
                print(">>> SIGNAL: HOLD <<<")
            
            print("-" * 60)
            
            # Sleep for timeframe duration
            time.sleep(timeframe_minutes * 60)
            
        except KeyboardInterrupt:
            print("\nStopping bot...")
            break
        except Exception as e:
            print(f"Error in prediction loop: {e}")
            print("Retrying in 1 minute...")
            time.sleep(60)


if __name__ == "__main__":
    main()

