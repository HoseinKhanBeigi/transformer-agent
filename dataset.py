"""
Custom PyTorch Dataset for price sequences
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple


class PriceDataset(Dataset):
    """Custom dataset for price sequences"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]

