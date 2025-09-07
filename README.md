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
- **Synthetic datasets**: Generated using Granser’s forward gravity model for training, validation, and testing.  
- **Field datasets (optional)**: Can be placed in the `data/field/` directory for real-case evaluation.

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Citation
If you use this code in your research, please cite our publication:


