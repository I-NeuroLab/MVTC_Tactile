# MVTC: Tactile Perception & Generation Framework

This repository provides the official implementation for **Tactile Perception (Regression/Classification)** and **Virtual Tactile Data Generation**. The project analyzes tactile signals using bio-inspired neural networks (TacNet) and generates synthetic tactile data using Conditional Variational Autoencoders (CVAE) with a split latent space.

## 📂 Repository Structure

The codebase is organized into two main projects. Based on the configuration, source codes (`src`) and model definitions (`models`) are separated.

```bash
.
├── Tactile-Library/
│   └── Everyday-Objects/           # [Task 1] Tactile Perception (TacNet)
│       ├── main_FB.py              # Main training script
│       ├── main_FB_loadmdl.py      # Inference/Evaluation script
│       ├── models/
│       │   └── TacNet_FB.py        # TacNet Model (Hybrid Transformer + FIR FilterBanks)
│       └── src/
│           ├── processes.py        # Training loops & Metrics (RMSE, R2, etc.)
│           ├── utils.py            # Utilities (EarlyStopping, Attention Rollout)
│           ├── arguments.py        # Argument parser configuration
│           └── Tactiledatasets_save.py # Dataset loader & Z-score normalization
│
└── Virtual-Tactile/
    └── Generation+CVAE+class/      # [Task 2] Virtual Tactile Generation (CVAE)
        ├── main_VAE_add+condition.py # Main training & generation script
        ├── models/
        │   └── VAE_add+condition.py  # VAE Model (Split Latent Space for Perception/Material)
        └── src/
            ├── processes_add_condition.py # Loss functions (Centroid Loss) & Training loops
            ├── arguments.py          # Argument parser configuration
            └── Tactiledatasets_save.py # Dataset loader for generation task


```

🛠️ Prerequisites
Python 3.8+

PyTorch (CUDA support recommended)

NumPy, SciPy, Pandas, Scikit-learn

```Bash
pip install torch numpy scipy pandas scikit-learn

