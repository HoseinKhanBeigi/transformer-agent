# 🚀 Deployment Guide - Vast.ai Training & Inference

## 📋 **Overview**

This guide shows you how to:
1. **Deploy to Vast.ai** for GPU training
2. **Download the trained model**
3. **Use the model locally** for live predictions (no retraining needed)

---

## 🎯 **Step 1: Deploy to Vast.ai for Training**

### **1.1 Rent a GPU Instance**

1. Go to https://cloud.vast.ai/
2. Search for GPU (recommended: RTX 3090 or RTX 4090)
3. Filter:
   - **GPU:** RTX 3090 (24GB) or better
   - **CUDA:** 11.8 or 12.x
   - **Disk:** 50GB+
   - **Internet:** Enabled
4. Click "Rent" and note your SSH details

### **1.2 Connect to Vast.ai Instance**

```bash
# SSH into your instance
ssh -p <port> root@<ip>
# Enter password when prompted
```

### **1.3 Upload Your Project**

**Option A: Using Git (Recommended)**
```bash
# On Vast.ai instance
cd /root
git clone <your-repo-url>
cd transformer-agent
```

**Option B: Using SCP (from your local machine)**
```bash
# From your local machine
scp -P <port> -r /path/to/transformer-agent root@<ip>:/root/
```

### **1.4 Setup Environment**

```bash
# On Vast.ai instance
cd /root/transformer-agent

# Run setup script
bash vast_ai_quick_setup.sh

# OR manually:
apt-get update
apt-get install -y python3 python3-pip git screen
pip3 install -r requirements.txt

# Verify GPU
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### **1.5 Train the Model**

```bash
# Use screen to keep training running if connection drops
screen -S training

# Run training
python3 crypto_trading_bot.py

# Detach: Press Ctrl+A, then D
# Reattach later: screen -r training
```

**What happens:**
1. Fetches 50,000 candles from Binance (~5-10 min)
2. Calculates 33 features (~2-3 min)
3. Trains model for 20 epochs (~15-30 min on GPU)
4. Saves model to `trained_model.pth`

**Expected output:**
```
Fetching 50000 candles...
...
Training samples: 39952
Epoch 1/20 - Train Loss: 1234.567 - Val Loss: 1235.123
...
Training completed!
Model saved to 'trained_model.pth'
```

### **1.6 Download Trained Model**

**From your local machine:**
```bash
# Download the trained model
scp -P <port> root@<ip>:/root/transformer-agent/trained_model.pth ./

# Optional: Download data files too
scp -P <port> root@<ip>:/root/transformer-agent/data/* ./data/
```

### **1.7 Stop Vast.ai Instance**

- Go to Vast.ai dashboard
- Click "Stop" on your instance (stops billing)

---

## 🎯 **Step 2: Use Trained Model Locally (Inference Only)**

### **2.1 Setup Local Environment**

```bash
# On your local machine
cd /path/to/transformer-agent

# Install dependencies (if not already)
pip install -r requirements.txt
```

### **2.2 Place Model File**

Make sure `trained_model.pth` is in the project directory:
```
transformer-agent/
├── trained_model.pth  ← Your trained model (from Vast.ai)
├── run_inference_only.py
├── crypto_trading_bot.py
└── ...
```

### **2.3 Run Inference Only**

```bash
# Run inference (no training, just predictions)
python run_inference_only.py
```

**What happens:**
1. Loads pre-trained model from `trained_model.pth`
2. Fetches latest 100 candles from Binance
3. Calculates features
4. Makes predictions every 15 minutes
5. Generates BUY/SELL/HOLD signals

**No training needed!** Just predictions. 🎉

---

## 📊 **Workflow Summary**

```
┌─────────────────────────────────────────┐
│ 1. TRAIN ON VAST.AI (One-time)         │
│    - Upload code to Vast.ai             │
│    - Train model (20-45 min)            │
│    - Download trained_model.pth         │
│    - Stop Vast.ai instance              │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. USE LOCALLY (Ongoing)                 │
│    - Place trained_model.pth locally     │
│    - Run: python run_inference_only.py   │
│    - Get live predictions every 15 min  │
│    - No GPU needed, runs on CPU fine    │
└─────────────────────────────────────────┘
```

---

## 💰 **Cost Breakdown**

### **Training (One-time):**
- Vast.ai GPU: ~$0.20-0.50 (20-45 minutes)
- **Total: ~$0.25**

### **Inference (Ongoing):**
- Local CPU: **FREE** (runs on your computer)
- Or keep Vast.ai running: ~$0.30/hour (not recommended)

**Recommendation:** Train once on Vast.ai, run inference locally for free!

---

## 🔧 **Troubleshooting**

### **Problem: "Model file not found"**
**Solution:**
```bash
# Make sure trained_model.pth is in the project directory
ls -la trained_model.pth
```

### **Problem: "CUDA out of memory" during training**
**Solution:**
```python
# In config.py, reduce batch size
BATCH_SIZE = 32  # or 16
```

### **Problem: "Missing features" error**
**Solution:**
- Make sure you're using the same feature configuration
- Check that `USE_EXTENDED_FEATURES` and `USE_FUNDAMENTAL_DATA` match training settings

### **Problem: Connection drops during training**
**Solution:**
```bash
# Use screen
screen -S training
python3 crypto_trading_bot.py
# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

---

## 📁 **File Structure**

```
transformer-agent/
├── crypto_trading_bot.py      # Full training + inference
├── run_inference_only.py      # Inference only (use this after training)
├── trained_model.pth          # Saved model (download from Vast.ai)
├── config.py                  # Configuration
├── data/                      # Historical data (optional)
│   └── BTC_USDT_15m_*.csv
└── ...
```

---

## 🎯 **Quick Commands**

### **On Vast.ai (Training):**
```bash
# Setup
bash vast_ai_quick_setup.sh

# Train
screen -S training
python3 crypto_trading_bot.py
# Ctrl+A, D to detach
```

### **On Local Machine (Inference):**
```bash
# Run inference
python run_inference_only.py
```

### **Download Model:**
```bash
# From local machine
scp -P <port> root@<ip>:/root/transformer-agent/trained_model.pth ./
```

---

## ✅ **Checklist**

### **Training on Vast.ai:**
- [ ] Rented GPU instance
- [ ] SSH'd into instance
- [ ] Uploaded project files
- [ ] Installed dependencies
- [ ] Verified GPU (nvidia-smi)
- [ ] Started training (screen session)
- [ ] Model saved (trained_model.pth exists)
- [ ] Downloaded model to local machine
- [ ] Stopped Vast.ai instance

### **Inference Locally:**
- [ ] trained_model.pth in project directory
- [ ] Dependencies installed locally
- [ ] Run: python run_inference_only.py
- [ ] Getting predictions every 15 minutes

---

## 🚀 **Summary**

1. **Train once on Vast.ai** (~$0.25, 20-45 min)
2. **Download model** to your computer
3. **Run inference locally** (free, forever)
4. **Get signals** every 15 minutes

**No need to retrain!** The model works for months. Only retrain if you want to update with new data.

---

## 💡 **Pro Tips**

1. **Save model regularly:** Model is saved automatically after training
2. **Keep same config:** Use same feature settings for training and inference
3. **Monitor performance:** Check if predictions are still accurate over time
4. **Retrain monthly:** Optional - retrain with fresh data monthly for best results
5. **Use screen:** Always use screen on Vast.ai to prevent disconnection

---

## 🎉 **You're Ready!**

Follow the steps above and you'll have:
- ✅ Trained model from Vast.ai GPU
- ✅ Running locally for free
- ✅ Live predictions every 15 minutes
- ✅ BUY/SELL/HOLD signals

Good luck! 🚀

