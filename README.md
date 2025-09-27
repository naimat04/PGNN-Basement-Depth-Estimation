# Gravity Inversion using PINN and CNN

This repository contains implementations of Physics-Informed Neural Networks (PINN) and Convolutional Neural Networks (CNN) for gravity anomaly to depth inversion, as described in our *Computer & Geosciences* publication.

## Repository Layout

### 🔹 PINN (Physics-Informed Neural Network)
- `main.py` – Main execution script  
- `model.py` – Neural network architecture + physics-informed loss  
- `physics.py` – Physics-based forward model (Granser’s method)  
- `data_loader.py` – Data loading and preprocessing  
- `utils.py` – Training utilities and callbacks  
- `config.py` – Configuration parameters  

### 🔹 CNN (Convolutional Neural Network)
- `main.py` – Main execution script  
- `model.py` – CNN architecture  
- `data_loader.py` – Data loading and preprocessing  
- `utils.py` – Training utilities and callbacks  
- `config.py` – Configuration parameters  

## Usage

### Train the PINN model
```bash
cd PINN
python main.py
```

### Train the CNN model
```bash
cd CNN
python main.py
```

## Dependencies
- TensorFlow 2.x  
- NumPy  
- SciPy  
- scikit-learn  

## Data
- Synthetic datasets: Generated using Granser’s forward gravity model for training, validation, and testing.
📥 [Download Dataset](https://zenodo.org/records/17071693?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI3YTdlZGMyLTM0YjQtNGI4Yi05NDZjLWQ3MjRjMTg0ZGNiOSIsImRhdGEiOnt9LCJyYW5kb20iOiJkYTVhODRjMGQ2NDJmZDIxZjhlMDlkNTcxOGU3NWFjZCJ9.xZ-yZCAJ97JS9WzQFH--qiEE9zEKdhwjagVaN82a0fvHlv61ME18kN7pRD11RaX4pybD_9fTGsVO3T6h4iRrxw)

- **Field datasets (optional)**: Can be placed in the `data/field/` directory for real-case evaluation.

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Citation
If you use this code in your research, please cite our publication:


