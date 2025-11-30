# Vast.ai Setup Guide for Crypto Trading Bot

## 🚀 Why Vast.ai is Perfect for This Project

- **Cost-Effective**: Much cheaper than AWS/Azure GPU instances
- **Fast Training**: GPU training is 10-50x faster than CPU
- **Pay-Per-Use**: Only pay for training time (typically $0.20-$1.00/hour)
- **Easy Setup**: Simple SSH connection, works like a regular Linux server

## 📋 Step-by-Step Setup

### 1. Create Vast.ai Account
1. Go to https://cloud.vast.ai/
2. Sign up and add payment method
3. Verify your account

### 2. Rent a GPU Instance

**Recommended GPU Options:**
- **RTX 3090** (24GB VRAM) - Best value, ~$0.30-0.50/hour
- **RTX 4090** (24GB VRAM) - Fastest, ~$0.50-0.80/hour
- **A100** (40GB VRAM) - Professional grade, ~$1.00-2.00/hour

**Minimum Requirements:**
- **8GB+ VRAM** (RTX 3060, RTX 3070 work fine)
- **Ubuntu 20.04/22.04** (most common)
- **Python 3.8+**

**Search Filters:**
- GPU: RTX 3090 or better
- CUDA: 11.8 or 12.x
- Disk: 50GB+ (for data and model)
- Internet: Enabled (for Binance API)

### 3. Connect to Your Instance

Once you rent an instance, you'll get:
- **SSH Command**: `ssh -p <port> root@<ip>`
- **Password**: Provided in dashboard

**Connect:**
```bash
ssh -p <port> root@<ip>
# Enter password when prompted
```

### 4. Setup Environment on Vast.ai

Once connected, run these commands:

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install Python and pip
apt-get install -y python3 python3-pip git

# Install CUDA toolkit (if not pre-installed)
# Most Vast.ai instances already have CUDA, check with:
nvidia-smi

# Clone or upload your project
# Option 1: If using git
git clone <your-repo-url>
cd transformer-agent

# Option 2: Upload files via SCP
# From your local machine:
# scp -P <port> -r /path/to/transformer-agent root@<ip>:/root/
```

### 5. Install Dependencies

```bash
cd transformer-agent

# Install Python packages
pip3 install -r requirements.txt

# Verify PyTorch CUDA support
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected Output:**
```
CUDA available: True
CUDA device: NVIDIA GeForce RTX 3090
```

### 6. Optimize for GPU Training

**Update `config.py` for GPU:**
```python
# Increase batch size for GPU (more memory available)
BATCH_SIZE = 128  # or 256 if you have 24GB+ VRAM

# Can train longer with faster GPU
EPOCHS = 30  # More epochs = better model (if not overfitting)
```

### 7. Run Training

```bash
# Run the bot (training happens first)
python3 crypto_trading_bot.py
```

**Monitor GPU Usage:**
```bash
# In another terminal, watch GPU usage
watch -n 1 nvidia-smi
```

## 💰 Cost Estimation

**Training Time:**
- Data fetching: ~5-10 minutes
- Feature engineering: ~2-3 minutes
- Model training (20 epochs): ~10-30 minutes (depends on GPU)
- **Total: ~20-45 minutes**

**Cost:**
- RTX 3090: $0.30/hour × 0.75 hours = **~$0.23**
- RTX 4090: $0.50/hour × 0.5 hours = **~$0.25**
- RTX 3060: $0.20/hour × 1 hour = **~$0.20**

**Very affordable!** 🎉

## 🔧 GPU Optimization Tips

### 1. Increase Batch Size
More GPU memory = larger batches = faster training:

```python
# In config.py
BATCH_SIZE = 128  # or 256 for 24GB+ GPUs
```

### 2. Use Mixed Precision Training
Faster training with less memory (optional enhancement):

```python
# In trainer.py, add:
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    predictions = model(sequences)
    loss = criterion(predictions, targets)
```

### 3. Monitor GPU Memory
```bash
# Check GPU memory usage
nvidia-smi

# Watch in real-time
watch -n 1 nvidia-smi
```

## 📊 Performance Comparison

| Device | Training Time (20 epochs) | Cost |
|--------|---------------------------|------|
| CPU (MacBook M1) | ~2-4 hours | Free |
| CPU (Intel i7) | ~4-8 hours | Free |
| RTX 3060 (8GB) | ~30-45 min | ~$0.15 |
| RTX 3090 (24GB) | ~15-25 min | ~$0.20 |
| RTX 4090 (24GB) | ~10-20 min | ~$0.25 |
| A100 (40GB) | ~8-15 min | ~$0.50 |

**GPU is 10-30x faster!** ⚡

## 🛠️ Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
```python
# Reduce batch size in config.py
BATCH_SIZE = 32  # or 16
```

### Issue: "CUDA not available"
**Solution:**
```bash
# Check CUDA installation
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Slow data fetching
**Solution:**
- Data fetching uses CPU/network, not GPU
- This is normal and expected
- GPU only speeds up model training

### Issue: Connection timeout
**Solution:**
- Vast.ai instances may disconnect after inactivity
- Use `screen` or `tmux` to keep sessions alive:
```bash
# Install screen
apt-get install -y screen

# Start screen session
screen -S training

# Run your script
python3 crypto_trading_bot.py

# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

## 📦 Save Your Trained Model

After training, save the model to avoid re-training:

```python
# Add to crypto_trading_bot.py after training:
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'feature_columns': feature_columns,
    'config': {
        'input_dim': len(feature_columns),
        'd_model': D_MODEL,
        # ... other config
    }
}, 'trained_model.pth')
```

**Download model:**
```bash
# From your local machine
scp -P <port> root@<ip>:/root/transformer-agent/trained_model.pth ./
```

## 🎯 Best Practices

1. **Start Small**: Test with fewer epochs first
2. **Monitor Costs**: Check Vast.ai dashboard regularly
3. **Save Checkpoints**: Save model after training
4. **Use Screen/Tmux**: Keep sessions alive
5. **Download Results**: Save model and logs before instance ends

## 🔄 Workflow Summary

```bash
# 1. Rent GPU on Vast.ai
# 2. SSH into instance
ssh -p <port> root@<ip>

# 3. Setup environment
cd transformer-agent
pip3 install -r requirements.txt

# 4. Run training
python3 crypto_trading_bot.py

# 5. (Optional) Save model
# Add model saving code, then:
python3 crypto_trading_bot.py

# 6. Download model (from local machine)
scp -P <port> root@<ip>:/root/transformer-agent/trained_model.pth ./

# 7. Stop instance on Vast.ai dashboard (to stop billing)
```

## ✅ Verification Checklist

- [ ] Vast.ai account created and verified
- [ ] GPU instance rented (RTX 3090 or better)
- [ ] SSH connection successful
- [ ] Python 3.8+ installed
- [ ] CUDA available (`nvidia-smi` works)
- [ ] PyTorch with CUDA installed
- [ ] Project files uploaded
- [ ] Dependencies installed
- [ ] GPU detected by PyTorch
- [ ] Training runs successfully
- [ ] Model saved (optional)

## 🎉 You're Ready!

Your bot will train **much faster** on Vast.ai GPU. The code already auto-detects CUDA, so it will automatically use the GPU when available!

**Estimated savings:** 
- CPU training: 2-4 hours
- GPU training: 15-30 minutes
- **Time saved: 1.5-3.5 hours!** ⚡

