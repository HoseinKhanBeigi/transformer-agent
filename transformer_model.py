"""
Transformer model architecture for price prediction
"""

import torch
import torch.nn as nn
from typing import Tuple


class TransformerModel(nn.Module):
    """Transformer model for price prediction"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 100
    ):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_length, d_model) * 0.1
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Decoder head
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            Predicted price tensor of shape (batch_size, 1)
        """
        batch_size, seq_length, _ = x.shape
        
        # Embed input
        x = self.input_embedding(x)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_length, :].unsqueeze(0)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
        
        # Global average pooling across time dimension
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Decoder head
        output = self.decoder(x)  # (batch_size, 1)
        
        return output

