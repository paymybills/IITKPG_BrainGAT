import json

kaggle_notebook_path = '/home/moew/Documents/ABIDE/kagglebooks/BrainGAT_Kaggle.ipynb'

# Bulletproof Import Cell
robust_import_cell = """# Imports & Setup (Self-Healing)
import os
import sys
import subprocess
import importlib

def install_dependencies():
    print(" Installing torch-geometric and dependencies...")
    # Use sys.executable to ensure we install to the current python environment
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch-geometric", "torch-scatter", "torch-sparse"])
    print(" Dependencies installed.")
    # Invalidate caches to find the new modules
    importlib.invalidate_caches()

try:
    import torch_geometric
except ImportError:
    install_dependencies()

# Re-import to be safe
import torch_geometric

import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

# Check for CUDA/GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   Note: Running on CPU - training will be slower")"""

def make_notebook_bulletproof():
    with open(kaggle_notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Find the cell that starts with "# Imports & Setup"
    import_cell_index = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and "# Imports & Setup" in "".join(cell['source']):
            import_cell_index = i
            break
            
    if import_cell_index == -1:
        # Fallback: It might be index 3 based on previous steps
        import_cell_index = 3
        
    print(f"Updating Import cell at index {import_cell_index}...")
    
    # Create new cell structure
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": robust_import_cell.splitlines(keepends=True)
    }
    
    # Update the notebook
    nb['cells'][import_cell_index] = new_cell
    
    with open(kaggle_notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Made notebook bulletproof: {kaggle_notebook_path}")

if __name__ == "__main__":
    make_notebook_bulletproof()
