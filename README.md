# Basement Depth Estimation from Gravity Data using PINNs and CNNs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the paper **"Basement Depth Estimation from Gravity Data Using Physics-Informed Neural Networks and its Comparison with Data-Driven Deep Learning"** (submitted to *Computers & Geosciences*). It provides code for comparing **Physics-Informed Neural Networks (PINN)** and **Data-Driven Convolutional Neural Networks (CNN)** for inverting gravity anomalies to estimate basement topography.

## 📖 Description

Estimating basement depth from gravity data is a classic geophysical inverse problem. This work implements and compares two deep learning approaches:

1. **Physics-Informed Neural Network (PINN):** A neural network trained with a hybrid loss function that combines data misfit with the governing physical equations (using Granser's forward model).
2. **Convolutional Neural Network (CNN):** A purely data-driven network trained on synthetic gravity-depth pairs.

The primary goal is to evaluate the potential of physics-constrained learning against traditional supervised learning for this geophysical task.

**Key Features:**
* Implementation of a PINN with a custom physics loss based on Granser's method
* Implementation of a comparative CNN baseline
* Scripts for training, validation, and testing on both synthetic and field data
* Utilities for visualizing results and metrics

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/naimat04/naimat04-PINN-Basement-Depth-Estimation.git
cd naimat04-PINN-Basement-Depth-Estimation
```

### 2. Create a Python Environment (Recommended)
Using conda:
```bash
conda create -n gravity_inversion python=3.9
conda activate gravity_inversion
```

Using venv:
```bash
python -m venv venv
# On Windows: venv\Scripts\activate
# On Linux/Mac: source venv/bin/activate
```

### 3. Install Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

## 📁 Repository Structure

```
.
├── PINN/                  # Physics-Informed Neural Network implementation
│   ├── config.py         # Configuration parameters (model, training, physics)
│   ├── model.py          # Neural network architecture & PINN loss definition
│   ├── physics.py        # Granser's forward model & physics computation
│   ├── data_loader.py    # Synthetic/field data loading and preprocessing
│   ├── utils.py          # Training callbacks, visualization, metrics
│   └── main.py           # Main training and evaluation script
│
├── CNN/                  # Convolutional Neural Network implementation
│   ├── config.py         # Configuration parameters
│   ├── model.py          # CNN architecture
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── utils.py          # Training utilities
│   └── main.py           # Main training and evaluation script
│
├── data/                 # Data directory (not in repo, see below)
│   ├── synthetic/        # Placeholder for synthetic data
│   └── field/           # Placeholder for field data (optional)
│
├── trained_models/      # Saved model weights (can be added via Git LFS or Zenodo)
├── results/             # Generated plots and output files
├── tests/               # Unit tests for critical functions
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 📊 Data Preparation

### Synthetic Data
The synthetic dataset used in the paper can be downloaded from Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17071693.svg)](https://doi.org/10.5281/zenodo.17071693)

**Steps to set up:**
1. Download the dataset (`synthetic_data.zip`) from the link above
2. Extract it into the `data/synthetic/` directory. The structure should look like:
   ```
   data/synthetic/
   ├── train/
   │   ├── gravity_profiles_train.npy
   │   └── depth_profiles_train.npy
   ├── val/
   │   ├── gravity_profiles_val.npy
   │   └── depth_profiles_val.npy
   └── test/
       ├── gravity_profiles_test.npy
       └── depth_profiles_test.npy
   ```

### Field Data (Optional)
To test with your own field data:
1. Place your gravity data (as a `.npy` or `.txt` file) in `data/field/`
2. Update the `data_path` in the relevant `config.py` file to point to your data
3. Ensure your data format matches the expected input shape (see `data_loader.py` for details)

## 🏃 Quick Start: Running a Test for Reviewers

To quickly verify the installation and run a minimal example, follow these steps. This script will load a small sample, train a model for a few epochs, and produce a test plot.

### For the PINN Model:
```bash
cd PINN
# This runs a short training and validation cycle on a small subset of data
python main.py --mode test --epochs 5
```
**Expected Output:** The script should start training, print loss values for 5 epochs, and save a sample prediction plot in the `../results/` directory.

### For the CNN Model:
```bash
cd CNN
# This runs a short training and validation cycle on a small subset of data
python main.py --mode test --epochs 5
```

> **Note for Reviewers:** The `--mode test` flag uses a pre-defined small dataset embedded in the code/repository to ensure a fast run (2-5 minutes). For full reproduction, please see the next section.

## 🔧 Full Reproduction of Paper Results

### 1. Train the PINN Model
To train the PINN model from scratch with the full dataset and default hyperparameters:
```bash
cd PINN
python main.py --mode train
```
Training logs, model checkpoints, and loss plots will be saved (typically in `./logs/` or `../results/`).

### 2. Train the CNN Model
Similarly, to train the CNN baseline:
```bash
cd CNN
python main.py --mode train
```

### 3. Evaluate a Pre-trained Model
To evaluate a saved model on the test set and generate final figures:
```bash
# For PINN
cd PINN
python main.py --mode evaluate --model_path ../trained_models/pinn_final.h5

# For CNN
cd CNN
python main.py --mode evaluate --model_path ../trained_models/cnn_final.h5
```

## 🧪 Testing

We include unit tests to verify the core functionality. Run them using pytest:
```bash
# Test the forward physics model
python -m pytest tests/test_physics.py -v

# Test the data loading utilities
python -m pytest tests/test_dataloader.py -v
```

## 🙏 Acknowledgments
* We thank the developers of TensorFlow and the core PINN research community
* This work was supported by Indian Institute of Technology Bombay 
* The forward modeling code is based on Granser's method (Granser, 1987)

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Important Notes for Users

1. **First-time Setup:** After cloning, ensure you create the required directories:
   ```bash
   mkdir -p data/synthetic data/field results trained_models
   ```

2. **Hardware Requirements:** 
   - Recommended: 16GB RAM, NVIDIA GPU with 4GB+ VRAM for faster training

3. **Troubleshooting:** If you encounter any issues, please check:
   - All dependencies are installed correctly (`pip install -r requirements.txt`)
   - The data is placed in the correct directory structure
   - You have sufficient disk space for saving models and results

## 🔗 Related Resources
* Granser, H. (1987). THREE‐DIMENSIONAL INTERPRETATION OF GRAVITY DATA FROM SEDIMENTARY BASINS USING AN EXPONENTIAL DENSITY‐DEPTH FUNCTION*. Geophysical Prospecting, 35(9), 1030–1041. https://doi.org/10.1111/j.1365-2478.1987.tb00858.x
