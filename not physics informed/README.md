# Physics-Informed Gravity Inversion

This repository contains the implementation of a physics-informed neural network for gravity field to depth inversion, as described in our Computer & Geosciences publication.

## File Structure
- `main.py` - Main execution script
- `model.py` - Neural network architecture and custom loss functions
- `physics.py` - Physics-based forward model (Granser's method)
- `data_loader.py` - Data loading and preprocessing
- `utils.py` - Training utilities and callbacks
- `config.py` - Configuration parameters
- `README.md` - This file

## Usage
Run `python main.py` to train the model

## Dependencies
- TensorFlow 2.x
- NumPy
- SciPy
- scikit-learn
