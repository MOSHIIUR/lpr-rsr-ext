# Google Colab Training Guide - LPR Super-Resolution

Complete guide for training the License Plate Recognition Super-Resolution model on Google Colab with free GPU.

## Table of Contents
1. [Why Use Google Colab?](#why-use-google-colab)
2. [Quick Setup](#quick-setup)
3. [Step-by-Step Guide](#step-by-step-guide)
4. [Colab Limitations](#colab-limitations)
5. [Tips & Best Practices](#tips--best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Why Use Google Colab?

**Benefits:**
- Free GPU access (NVIDIA T4, 15GB VRAM)
- No local setup required
- Pre-installed Python packages
- Easy to share notebooks
- Automatic environment management

**Ideal for:**
- Users without local GPU
- Quick experiments
- Sharing reproducible results
- Learning and prototyping

**Limitations:**
- 12-hour session timeout (see workarounds below)
- Limited disk space (~100GB)
- May disconnect during long training
- Need to save checkpoints to Google Drive

---

## Quick Setup

### 1. Create a New Colab Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click "New Notebook"
3. Enable GPU: `Runtime → Change runtime type → GPU → Save`

### 2. Run This Setup Code

```python
# Cell 1: Enable GPU and check
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 3. Clone Repository and Setup

```python
# Cell 2: Clone repository
!git clone https://github.com/yourusername/lpr-rsr-ext.git
%cd lpr-rsr-ext/Proposed
```

### 4. Install Dependencies

```python
# Cell 3: Install required packages
!pip install -q tensorflow==2.15.1
!pip install -q tf-keras  # Keras 2 compatibility layer for loading old OCR models
!pip install -q albumentations==1.3.0
!pip install -q pandas tqdm scikit-learn

# Verify installations
import tensorflow as tf
import tf_keras
print(f"TensorFlow: {tf.__version__}")
print(f"tf-keras (Keras 2): {tf_keras.__version__}")
```

### 5. Mount Google Drive

```python
# Cell 4: Mount Google Drive for checkpoints
from google.colab import drive
drive.mount('/content/drive')

# Create checkpoint directory in Drive
!mkdir -p /content/drive/MyDrive/lpr_checkpoints
```

### 6. Prepare Dataset

**Option A: Use Pre-prepared dataset.txt**
```python
# Cell 5a: Upload dataset.txt
from google.colab import files
uploaded = files.upload()  # Upload your dataset.txt
!mkdir -p ../dataset
!mv dataset.txt ../dataset/
```

**Option B: Link Dataset from Drive (Recommended)**
```python
# Cell 5b: Link dataset from Google Drive
# First, upload your dataset to Drive: MyDrive/lpr_dataset/
!ln -s /content/drive/MyDrive/lpr_dataset /content/lpr-rsr-ext/dataset
!ls ../dataset/  # Verify
```

**Option C: Prepare Dataset from ICPR Format**
```python
# Cell 5c: Prepare dataset using prepare_dataset.py
# First, upload raw dataset to Drive: MyDrive/raw_dataset/Brazilian/
!python ../prepare_dataset.py \
    --dataset-root /content/drive/MyDrive/raw_dataset/Brazilian \
    --output-dir ../dataset \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15

# Verify
!ls ../dataset/
!head -5 ../dataset/dataset.txt
```

### 7. Upload OCR Models

```python
# Cell 6: Upload OCR models (one-time setup)
# Create folder structure
!mkdir -p ../saved_models/RodoSol-SR
!mkdir -p ../saved_models/PKU-SR

# Upload model files to Google Drive first, then:
!cp /content/drive/MyDrive/ocr_models/RodoSol-SR/* ../saved_models/RodoSol-SR/
!cp /content/drive/MyDrive/ocr_models/PKU-SR/* ../saved_models/PKU-SR/

# Verify
!ls -lh ../saved_models/RodoSol-SR/
```

### 8. Test with Debug Mode

```python
# Cell 7: Quick test with debug mode
!python training.py \
    -t ../dataset/dataset.txt \
    -s /content/drive/MyDrive/lpr_checkpoints \
    -b 8 \
    -m 0 \
    --debug \
    --debug-samples 20
```

### 9. Start Full Training

```python
# Cell 8: Full training (saves to Google Drive)
!python training.py \
    -t ../dataset/dataset.txt \
    -s /content/drive/MyDrive/lpr_checkpoints \
    -b 16 \
    -m 0
```

---

## Step-by-Step Guide

### Step 1: Prepare Your Files Locally

Before using Colab, prepare these files locally:

1. **Dataset file** (`dataset.txt`)
   - Format: `HR_path;LR_path;split`
   - Or upload actual images to Google Drive

2. **OCR Models**
   - Download RodoSol-SR and PKU-SR
   - Upload to Google Drive: `MyDrive/ocr_models/`

3. **Images** (if using actual images)
   - Upload to Google Drive: `MyDrive/lpr_dataset/`
   - Maintain directory structure

### Step 2: Create Colab Notebook

Create a new notebook with these cells:

#### Cell 1: Environment Check
```python
# Check GPU availability
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ GPU not enabled! Go to Runtime → Change runtime type → GPU")
```

#### Cell 2: Clone Repository
```python
# Clone the repository
!git clone https://github.com/yourusername/lpr-rsr-ext.git
%cd lpr-rsr-ext/Proposed

# Verify location
!pwd
!ls
```

#### Cell 3: Install Dependencies
```python
# Install required packages
!pip install -q --upgrade pip
!pip install -q tensorflow==2.15.1
!pip install -q tf-keras  # Keras 2 compatibility layer for loading old OCR models
!pip install -q opencv-python-headless  # headless version for Colab
!pip install -q albumentations==1.3.0
!pip install -q pandas tqdm scikit-learn

# Verify
import tensorflow as tf
import tf_keras
import cv2
import albumentations as A
print("✓ All packages installed")
print(f"TensorFlow: {tf.__version__}")
print(f"tf-keras (Keras 2): {tf_keras.__version__}")
print(f"OpenCV: {cv2.__version__}")
```

#### Cell 4: Mount Google Drive
```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Create directory structure
!mkdir -p /content/drive/MyDrive/lpr_checkpoints
!mkdir -p /content/drive/MyDrive/lpr_results
print("✓ Google Drive mounted")
```

#### Cell 5: Setup Dataset
```python
# Option 1: Link dataset from Drive
!ln -s /content/drive/MyDrive/lpr_dataset /content/lpr-rsr-ext/dataset
print("Dataset linked from Drive")

# Option 2: Upload dataset.txt only (if images are already in correct paths)
from google.colab import files
# Uncomment to upload:
# uploaded = files.upload()
# !mkdir -p ../dataset
# !mv dataset.txt ../dataset/

# Verify dataset
!head -5 ../dataset/dataset.txt
```

#### Cell 6: Setup OCR Models
```python
# Copy OCR models from Drive
!mkdir -p ../saved_models/RodoSol-SR
!mkdir -p ../saved_models/PKU-SR

!cp /content/drive/MyDrive/ocr_models/RodoSol-SR/* ../saved_models/RodoSol-SR/ 2>/dev/null || echo "RodoSol-SR not found in Drive"
!cp /content/drive/MyDrive/ocr_models/PKU-SR/* ../saved_models/PKU-SR/ 2>/dev/null || echo "PKU-SR not found in Drive"

# Verify
!ls -lh ../saved_models/RodoSol-SR/
!ls -lh ../saved_models/PKU-SR/
```

#### Cell 7: Debug Mode Test
```python
# Quick test with 20 samples
!python training.py \
    -t ../dataset/dataset.txt \
    -s /content/drive/MyDrive/lpr_checkpoints/debug \
    -b 8 \
    -m 0 \
    --debug \
    --debug-samples 20

print("\n✓ Debug test completed successfully!")
```

#### Cell 8: Full Training
```python
# Full training - saves checkpoints to Google Drive
!python training.py \
    -t ../dataset/dataset.txt \
    -s /content/drive/MyDrive/lpr_checkpoints \
    -b 16 \
    -m 0
```

#### Cell 9: Monitor GPU Usage (Optional)
```python
# Run this in a separate cell while training
!nvidia-smi
```

#### Cell 10: Resume Training (If Disconnected)
```python
# Resume from last checkpoint
!python training.py \
    -t ../dataset/dataset.txt \
    -s /content/drive/MyDrive/lpr_checkpoints \
    -b 16 \
    -m 1 \
    --model /content/drive/MyDrive/lpr_checkpoints/backup.pt
```

#### Cell 11: Test Trained Model
```python
# Test the trained model
!python testing.py \
    -t ../dataset/dataset.txt \
    -s /content/drive/MyDrive/lpr_results \
    -b 8 \
    --model /content/drive/MyDrive/lpr_checkpoints/backup.pt

# Download results
from google.colab import files
!zip -r results.zip /content/drive/MyDrive/lpr_results
files.download('results.zip')
```

---

## Colab Limitations

### 1. Session Timeout (12 Hours)

**Problem:** Colab disconnects after 12 hours of inactivity.

**Solutions:**

**A. Save checkpoints to Google Drive** (Recommended)
```python
# Always use Drive path for checkpoints
-s /content/drive/MyDrive/lpr_checkpoints
```

**B. Keep connection alive** (JavaScript)
```javascript
// Run this in browser console (F12)
function ClickConnect(){
    console.log("Clicking");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```

**C. Use Colab Pro**
- $10/month
- Longer runtimes (24 hours)
- Better GPU availability

### 2. Disk Space (Limited)

**Available:** ~100GB in `/content/`

**Solutions:**
- Store large files in Google Drive
- Delete intermediate files
- Use smaller dataset for testing
- Clean up after training:
  ```python
  !rm -rf /content/lpr-rsr-ext/.git  # Remove git history
  !rm -rf /content/lpr-rsr-ext/dataset/images  # If copied locally
  ```

### 3. GPU Availability

**Problem:** Sometimes GPU quota is exhausted.

**Solutions:**
- Try later (peak hours: 9am-5pm EST)
- Use Colab Pro for priority access
- Switch to CPU temporarily:
  ```python
  # Force CPU training
  device = torch.device('cpu')
  ```

### 4. Disconnection During Training

**Problem:** Browser closes or internet drops.

**Solutions:**
- Checkpoints saved to Drive persist across sessions
- Resume training with mode 1:
  ```python
  !python training.py ... -m 1 --model /content/drive/MyDrive/lpr_checkpoints/backup.pt
  ```
- Use `nohup` (limited effectiveness in Colab)

---

## Tips & Best Practices

### 1. Optimize Batch Size for Colab GPU

**Colab T4 GPU (15GB VRAM):**
```python
-b 16  # Good balance
-b 24  # If memory allows
-b 32  # Maximum (may OOM)
```

Test with debug mode first:
```python
!python training.py ... -b 16 --debug --debug-samples 20
```

### 2. Monitor Training Progress

**Install tensorboard:**
```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/lpr_checkpoints
```

**Check GPU usage:**
```python
# Run periodically
!nvidia-smi
```

**View loss in real-time:**
```python
# In a separate cell while training runs
!tail -f /content/lpr-rsr-ext/Proposed/training.log
```

### 3. Save Outputs to Drive

Always save to Drive, not `/content/`:

```python
# ✓ Good - persists across sessions
-s /content/drive/MyDrive/lpr_checkpoints

# ✗ Bad - lost on disconnect
-s ./checkpoints
```

### 4. Use Persistent Dataset Storage

**Upload dataset to Drive once:**
```python
# Organize in Drive:
MyDrive/
  ├── lpr_dataset/
  │   ├── dataset.txt
  │   ├── HR/
  │   └── LR/
  ├── ocr_models/
  │   ├── RodoSol-SR/
  │   └── PKU-SR/
  └── lpr_checkpoints/
```

**Link in Colab:**
```python
!ln -s /content/drive/MyDrive/lpr_dataset ../dataset
!ln -s /content/drive/MyDrive/ocr_models ../saved_models
```

### 5. Minimize Upload/Download

**Avoid:**
- Uploading large files every session
- Downloading large checkpoints repeatedly

**Instead:**
- Keep everything in Google Drive
- Only download final results

### 6. Test Locally First

Before using Colab:
1. Test with debug mode locally
2. Verify dataset format
3. Ensure OCR models load
4. Then replicate in Colab

### 7. Automate Setup with Script

Create `setup_colab.py`:
```python
# setup_colab.py
import os
from pathlib import Path

def setup_colab():
    """Automate Colab environment setup"""

    # Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # Create directories
    os.makedirs('/content/drive/MyDrive/lpr_checkpoints', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/lpr_results', exist_ok=True)

    # Link dataset and models
    if not os.path.exists('../dataset'):
        os.symlink('/content/drive/MyDrive/lpr_dataset', '../dataset')
    if not os.path.exists('../saved_models'):
        os.symlink('/content/drive/MyDrive/ocr_models', '../saved_models')

    # Verify GPU
    import torch
    if not torch.cuda.is_available():
        print("⚠️ WARNING: GPU not available!")
        return False

    print("✓ Colab environment ready")
    return True

# Run setup
setup_colab()
```

### 8. Use Colab Forms for Parameters

Make your notebook interactive:
```python
#@title Training Configuration
batch_size = 16  #@param {type:"slider", min:2, max:32, step:2}
debug_mode = False  #@param {type:"boolean"}
debug_samples = 20  #@param {type:"integer"}
checkpoint_path = "/content/drive/MyDrive/lpr_checkpoints"  #@param {type:"string"}

# Run training with form values
cmd = f"python training.py -t ../dataset/dataset.txt -s {checkpoint_path} -b {batch_size} -m 0"
if debug_mode:
    cmd += f" --debug --debug-samples {debug_samples}"

!{cmd}
```

---

## Troubleshooting

### GPU Not Available

```
⚠️ CUDA not available, training on CPU
```

**Solutions:**
1. Check runtime type: `Runtime → Change runtime type → GPU`
2. Restart runtime: `Runtime → Restart runtime`
3. Try later (GPU quota may be exhausted)
4. Use Colab Pro for guaranteed GPU

### Out of Memory on GPU

```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce batch size
-b 8  # Try 8 instead of 16
-b 4  # Or 4 if still failing

# Clear cache
import torch
torch.cuda.empty_cache()

# Restart runtime and try again
```

### Drive Mount Fails

```
Drive Mount Failed
```

**Solutions:**
1. Check Google account permissions
2. Try manual mount:
   ```python
   from google.colab import drive
   drive.mount('/content/drive', force_remount=True)
   ```
3. Use browser in incognito mode
4. Clear browser cookies

### Files Not Found After Disconnect

**Problem:** Files in `/content/` are lost.

**Solution:** Always use Google Drive paths:
```python
# ✓ Persists
/content/drive/MyDrive/lpr_checkpoints/backup.pt

# ✗ Lost on disconnect
/content/lpr-rsr-ext/checkpoints/backup.pt
```

### Slow Training

**Check GPU utilization:**
```python
!nvidia-smi
# GPU-Util should be >80%
```

**If low utilization:**
1. Increase batch size
2. Check data loading (shouldn't be bottleneck)
3. Verify GPU is being used:
   ```python
   import torch
   print(next(model.parameters()).device)  # Should show 'cuda:0'
   ```

### Session Keeps Disconnecting

**Solutions:**
1. Keep browser tab active
2. Use JavaScript keep-alive (see above)
3. Check internet connection
4. Avoid peak hours
5. Use Colab Pro for longer sessions

---

## Complete Example Notebook

Here's a complete, ready-to-run Colab notebook structure:

```python
# =============================================================================
# CELL 1: Environment Check
# =============================================================================
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# CELL 2: Setup
# =============================================================================
# Clone repo
!git clone https://github.com/yourusername/lpr-rsr-ext.git
%cd lpr-rsr-ext/Proposed

# Install dependencies
!pip install -q tensorflow==2.15.1 tf-keras albumentations opencv-python-headless

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# =============================================================================
# CELL 3: Link Data
# =============================================================================
!mkdir -p /content/drive/MyDrive/lpr_checkpoints
!ln -s /content/drive/MyDrive/lpr_dataset ../dataset
!ln -s /content/drive/MyDrive/ocr_models ../saved_models

# Verify
!ls ../dataset/
!ls ../saved_models/RodoSol-SR/

# =============================================================================
# CELL 4: Debug Test
# =============================================================================
!python training.py \
    -t ../dataset/dataset.txt \
    -s /content/drive/MyDrive/lpr_checkpoints/debug \
    -b 8 \
    -m 0 \
    --debug --debug-samples 20

# =============================================================================
# CELL 5: Full Training
# =============================================================================
!python training.py \
    -t ../dataset/dataset.txt \
    -s /content/drive/MyDrive/lpr_checkpoints \
    -b 16 \
    -m 0

# =============================================================================
# CELL 6: Test Model
# =============================================================================
!python testing.py \
    -t ../dataset/dataset.txt \
    -s /content/drive/MyDrive/lpr_results \
    -b 8 \
    --model /content/drive/MyDrive/lpr_checkpoints/backup.pt

# =============================================================================
# CELL 7: Download Results
# =============================================================================
from google.colab import files
!zip -r results.zip /content/drive/MyDrive/lpr_results
files.download('results.zip')
```

---

## Additional Resources

- [Google Colab Documentation](https://colab.research.google.com/notebooks/welcome.ipynb)
- [Colab Pro](https://colab.research.google.com/signup) - $10/month for better GPUs
- See `QUICK_START.md` for local setup
- See `TRAINING_GUIDE.md` for training details

---

**Happy Training on Colab!** 🚀
