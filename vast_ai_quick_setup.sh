#!/bin/bash
# Quick setup script for Vast.ai GPU instances
# Run this after SSH'ing into your Vast.ai instance

echo "🚀 Setting up Crypto Trading Bot on Vast.ai GPU..."

# Update system
echo "📦 Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq

# Install Python and essentials
echo "🐍 Installing Python and dependencies..."
apt-get install -y python3 python3-pip git screen -qq

# Check GPU
echo "🎮 Checking GPU..."
nvidia-smi

# Install Python packages
echo "📚 Installing Python packages..."
pip3 install -q --upgrade pip
pip3 install -q -r requirements.txt

# Verify PyTorch CUDA
echo "✅ Verifying PyTorch CUDA support..."
python3 -c "import torch; print(f'\n🎯 CUDA Available: {torch.cuda.is_available()}'); print(f'🎯 GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}\n')"

echo "✨ Setup complete! You can now run:"
echo "   python3 crypto_trading_bot.py"
echo ""
echo "💡 Tip: Use 'screen' to keep training running:"
echo "   screen -S training"
echo "   python3 crypto_trading_bot.py"
echo "   (Press Ctrl+A, then D to detach)"

