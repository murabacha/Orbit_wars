# Orbit Wars Training on Google Colab 🚀

**Complete guide to train Orbit Wars AI on free GPU**

---

## ✅ Step 1: Prepare Your Code for Colab

### Option A: Upload to GitHub (Recommended)
```bash
# In your local machine:
cd /home/roba/Projects/Reinforcement_learning
git init
git add Orbit_wars/
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/orbit-wars-rl.git
git push -u origin main
```

Then in Colab, clone it:
```python
!git clone https://github.com/YOUR_USERNAME/orbit-wars-rl.git
%cd orbit-wars-rl/Orbit_wars
```

### Option B: Upload ZIP to Google Drive (Simpler)
1. Zip your `Orbit_wars/` folder locally
2. Upload to Google Drive root
3. In Colab:
```python
from google.colab import drive
drive.mount('/content/gdrive')
import shutil
shutil.unpack_archive('/content/gdrive/My Drive/Orbit_wars.zip', '/content/Orbit_wars')
%cd /content/Orbit_wars
```

---

## ✅ Step 2: Create Colab Notebook

**Go to:** https://colab.research.google.com/

**Create new notebook** and paste this code:

```python
# ============================================================
# ORBIT WARS TRAINING ON COLAB
# ============================================================

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/gdrive')

# Setup paths
import os
os.chdir('/content/gdrive/My Drive/Orbit_wars')  # Or your project path

# Install dependencies
!pip install -q torch numpy kaggle-environments gymnasium matplotlib pandas tensorboard -U

# Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ============================================================
# PHASE 2: BEHAVIORAL CLONING
# ============================================================

# Create data directory
!mkdir -p data/bc_dataset checkpoints logs

# Option: Generate BC data (100k rollouts from heuristic)
# This would take ~30 min. For now, we'll skip and focus on Phase 3 PPO.
# Uncomment below to run:
# !python training/generate_bc_data.py --num_rollouts 100000 --save_dir data/bc_dataset

# ============================================================
# PHASE 3: PPO TRAINING
# ============================================================

# Run PPO training
!python training/train_ppo.py \
  --total_timesteps 500000 \
  --opponent baseline \
  --batch_size 32 \
  --device cuda \
  --save_interval 50000 \
  --log_dir logs/ppo_training

# Monitor with TensorBoard (in Colab)
%load_ext tensorboard
%tensorboard --logdir logs/ppo_training

# ============================================================
# EVALUATION
# ============================================================

# Run evaluation
!python training/evaluation.py \
  --agent_path checkpoints/ppo_500k.pt \
  --num_games 50 \
  --save_results logs/eval_results.txt

# ============================================================
# SAVE RESULTS TO GOOGLE DRIVE
# ============================================================

import shutil
shutil.copy('logs/eval_results.txt', '/content/gdrive/My Drive/eval_results.txt')
shutil.copy('checkpoints/ppo_500k.pt', '/content/gdrive/My Drive/ppo_500k.pt')
print("✅ Results saved to Google Drive!")
```

---

## ✅ Step 3: Expected Output

```
GPU Available: True
GPU Name: NVIDIA A100-PCIE-40GB

Starting PPO training on cuda...
Epoch 1/10: loss=0.234, value_loss=0.156, entropy=1.234
Epoch 2/10: loss=0.198, value_loss=0.142, entropy=1.201
...
Evaluating checkpoints/ppo_500k.pt over 50 games...
Win Rate: 65.2%
✅ Results saved to Google Drive!
```

---

## ⚠️ Important Notes

### Colab Session Timeout
- Free Colab: **12 hours** max per session
- Pro Colab: **24 hours**
- **Solution:** For Phase 4 (self-play), split into multiple notebook runs or use Colab Pro

### Google Drive Storage
- Free: 15GB (should be enough)
- Save checkpoints frequently to Drive

### Bandwidth
- Training logs synced to Drive automatically
- Download final model before session expires

---

## 🔄 Workflow for Multi-Day Training

If training Phase 4 (5M steps = 12+ hours):

### Session 1 (Day 1):
```python
!python training/train_ppo.py \
  --total_timesteps 1000000 \
  --checkpoint_load checkpoints/ppo_500k.pt \
  --save_interval 100000
```
→ Save final checkpoint to Drive

### Session 2 (Day 2):
```python
!python training/train_ppo.py \
  --total_timesteps 5000000 \
  --checkpoint_load /content/gdrive/My\ Drive/ppo_1m.pt \
  --save_interval 100000
```

---

## 📋 Checklist for Colab

- [ ] Code uploaded to GitHub or Drive
- [ ] Colab notebook created
- [ ] GPU enabled (Runtime → Change Runtime Type → GPU)
- [ ] Google Drive mounted
- [ ] Dependencies installed
- [ ] Phase 2 or Phase 3 training started
- [ ] TensorBoard monitoring set up
- [ ] Checkpoints auto-saving to Drive
- [ ] Evaluation running successfully

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Out of memory** | Reduce `batch_size` from 32 → 16 |
| **CUDA OOM** | Clear cache: `torch.cuda.empty_cache()` |
| **Import errors** | Reinstall: `!pip install --upgrade kaggle-environments` |
| **Session timeout** | Save checkpoint every 10 mins to Drive |
| **GPU not available** | Restart runtime, verify GPU in settings |

---

## 🚀 Quick Start (3-Step Colab)

1. **Open:** https://colab.research.google.com/
2. **Upload:** Your `Orbit_wars.zip` to Google Drive
3. **Paste & Run:** The notebook code above

**Expected completion:** 2-4 hours for full Phase 3 PPO training

---

**Next Step:** Open a Colab notebook and run the training! 🎯
