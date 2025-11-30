"""
Training utilities for the transformer model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import LEARNING_RATE, DEVICE


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device = DEVICE
) -> None:
    """
    Train the transformer model
    
    Args:
        model: Transformer model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        device: Device to train on
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.to(device)
    model.train()
    
    print("\nStarting training...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device).unsqueeze(1)
                
                predictions = model(sequences)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.6f} - "
            f"Val Loss: {avg_val_loss:.6f}"
        )
    
    print("Training completed!\n")

