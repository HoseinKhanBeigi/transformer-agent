"""
Cryptocurrency Trading Bot with Transformer Model
Main entry point - orchestrates data fetching, training, and live prediction
"""

import time
import warnings
from datetime import datetime

import ccxt
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Import modular components
from config import (
    SYMBOL, TIMEFRAME, TRAINING_LIMIT, LIVE_UPDATE_LIMIT,
    SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, DEVICE,
    D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
    BINANCE_API_KEY, BINANCE_SECRET
)
from transformer_model import TransformerModel
from data_fetcher import fetch_massive_history, fetch_latest_update
from feature_engineering import calculate_features, get_feature_columns, prepare_sequences
from dataset import PriceDataset
from trainer import train_model

warnings.filterwarnings('ignore')

print(f"Using device: {DEVICE}")


def main():
    """Main execution function"""
    
    # Initialize exchange
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot'
        }
    })
    
    # Step 1: Fetch massive history and train model
    print("=" * 60)
    print("STEP 1: Fetching historical data and training model")
    print("=" * 60)
    
    try:
        df_history = fetch_massive_history(
            exchange,
            SYMBOL,
            TIMEFRAME,
            TRAINING_LIMIT
        )
    except Exception as e:
        print(f"Failed to fetch historical data: {e}")
        return
    
    # Calculate features
    print("\nCalculating features...")
    df_features = calculate_features(df_history)
    
    # Get feature columns
    feature_columns = get_feature_columns(df_features)
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    # Prepare sequences
    print("\nPreparing sequences...")
    sequences, targets = prepare_sequences(
        df_features,
        SEQUENCE_LENGTH,
        feature_columns
    )
    
    print(f"Created {len(sequences)} sequences")
    
    # Scale features
    print("\nScaling features...")
    scaler = RobustScaler()
    sequences_scaled = scaler.fit_transform(
        sequences.reshape(-1, sequences.shape[-1])
    ).reshape(sequences.shape)
    
    # Split data
    train_sequences, val_sequences, train_targets, val_targets = train_test_split(
        sequences_scaled,
        targets,
        test_size=0.2,
        shuffle=False  # Don't shuffle time series data
    )
    
    # Create datasets and loaders
    train_dataset = PriceDataset(train_sequences, train_targets)
    val_dataset = PriceDataset(val_sequences, val_targets)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Initialize model
    model = TransformerModel(
        input_dim=len(feature_columns),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_length=SEQUENCE_LENGTH
    )
    
    # Train model
    train_model(model, train_loader, val_loader, EPOCHS, DEVICE)
    
    # Save trained model
    print("Saving trained model...")
    import pickle
    model_save_data = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_columns': feature_columns,
        'config': {
            'input_dim': len(feature_columns),
            'd_model': D_MODEL,
            'nhead': NHEAD,
            'num_layers': NUM_LAYERS,
            'dim_feedforward': DIM_FEEDFORWARD,
            'dropout': DROPOUT,
            'max_seq_length': SEQUENCE_LENGTH
        }
    }
    torch.save(model_save_data, 'trained_model.pth')
    print("Model saved to 'trained_model.pth'\n")
    
    # Step 2: Switch to evaluation mode
    model.eval()
    print("Model switched to evaluation mode\n")
    
    # Step 3: Live prediction loop
    print("=" * 60)
    print("STEP 3: Starting live prediction loop")
    print("=" * 60)
    print(f"Monitoring {SYMBOL} on {TIMEFRAME} timeframe")
    print(f"Sleep interval: 15 minutes\n")
    
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
            
            # Update Fear & Greed Index with fresh data for live prediction (if enabled)
            from config import USE_FUNDAMENTAL_DATA
            if USE_FUNDAMENTAL_DATA:
                from fundamental_data import add_fear_greed_to_df
                df_live_features = add_fear_greed_to_df(df_live_features, use_current=False)
            
            # Check if we have enough data
            if len(df_live_features) < SEQUENCE_LENGTH:
                print(f"Not enough data. Have {len(df_live_features)}, need {SEQUENCE_LENGTH}")
                time.sleep(15 * 60)
                continue
            
            # Verify all required features exist
            missing_features = [col for col in feature_columns if col not in df_live_features.columns]
            if missing_features:
                print(f"Warning: Missing features {missing_features}. Skipping this iteration.")
                time.sleep(15 * 60)
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
            
            # Calculate percentage change
            price_change_pct = ((prediction - current_price) / current_price) * 100
            
            # Generate signal
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Current Price: ${current_price:.2f}")
            print(f"Predicted Price: ${prediction:.2f}")
            print(f"Expected Change: {price_change_pct:.2f}%")
            
            if price_change_pct > 0.1:
                print(">>> SIGNAL: STRONG BUY <<<")
            elif price_change_pct < -0.1:
                print(">>> SIGNAL: STRONG SELL <<<")
            else:
                print(">>> SIGNAL: HOLD <<<")
            
            print("-" * 60)
            
            # Sleep for 15 minutes (900 seconds)
            time.sleep(15 * 60)
            
        except KeyboardInterrupt:
            print("\nStopping bot...")
            break
        except Exception as e:
            print(f"Error in prediction loop: {e}")
            print("Retrying in 1 minute...")
            time.sleep(60)


if __name__ == "__main__":
    main()
