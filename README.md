Gravity Inversion using PINN and CNN
This repository contains the implementation of Physics-Informed Neural Networks (PINN) and Convolutional Neural Networks (CNN) for gravity anomaly to depth inversion, as described in our Computer & Geosciences publication.

├── PINN/                     # Physics-Informed Neural Network implementation
│   ├── main.py               # Main execution script
│   ├── model.py              # Neural network architecture + physics-informed loss
│   ├── physics.py            # Physics-based forward model (Granser's method)
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── utils.py              # Training utilities and callbacks
│   ├── config.py             # Configuration parameters
│   └── README.md             # Instructions for PINN usage
├── CNN/                      # Conventional CNN implementation
│   ├── main.py               # Main execution script
│   ├── model.py              # CNN architecture
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── utils.py              # Training utilities and callbacks
│   ├── config.py             # Configuration parameters
│   └── README.md             # Instructions for CNN usage
├── data/                     # Datasets for training and testing
│   ├── synthetic/            # Synthetic gravity anomaly and depth datasets
│   │   ├── train_data.npy
│   │   ├── valid_data.npy
│   │   └── test_data.npy
│   └── field/                # (Optional) Example field datasets
└── README.md                 # This file

Usage
Training the PINN Model
bash
cd PINN
python main.py
Training the CNN Model
bash
cd CNN
python main.py
Dependencies
TensorFlow 2.x

NumPy

SciPy

scikit-learn

Data
The data

Synthetic datasets generated using Granser's forward gravity model for training, validation, and testing

Field datasets (optional) can be placed in the data/field/ directory for real-case evaluation
