#!/bin/bash
# CUDA Brain GNN Environment Setup Script
# Creates a virtual environment with PyTorch + CUDA + PyTorch Geometric

echo "🧠 Setting up CUDA-compatible Brain GNN environment..."

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "❌ Python 3.12 not found. Please install Python 3.12 first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3.12 -m venv .venv-brain-gnn

# Activate environment
echo "🔄 Activating environment..."
source .venv-brain-gnn/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
echo "🔥 Installing PyTorch with CUDA 12.1..."
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
echo "🌐 Installing PyTorch Geometric..."
pip install torch-geometric==2.6.1

# Install PyG dependencies with CUDA support
echo "⚡ Installing PyG CUDA extensions..."
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Install data science packages
echo "📊 Installing data science packages..."
pip install numpy==2.1.2 pandas==2.3.2 scipy==1.16.2 scikit-learn==1.7.2
pip install matplotlib==3.10.6 seaborn==0.13.2 networkx==3.3

# Install Jupyter for notebook support
echo "📓 Installing Jupyter..."
pip install jupyter ipykernel ipython

# Install utilities
echo "🛠️ Installing utilities..."
pip install tqdm requests

# Test installation
echo "🧪 Testing CUDA installation..."
python -c "
import torch
import torch_geometric
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
print(f'✅ PyTorch Geometric: {torch_geometric.__version__}')
print(f'✅ Number of GPUs: {torch.cuda.device_count()}')
"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "To use this environment:"
echo "  source .venv-brain-gnn/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"