readme:
  title: "Basement Depth Estimation from Gravity Data using PINNs and CNNs"
  description: |
    This repository contains the code used in the paper:

    **Basement Depth Estimation from Gravity Data Using Physics-Informed Neural Networks and its Comparison with Data-Driven Deep Learning**

    The repository implements:

    • Physics-Informed Neural Network (PINN)  
    • Data-driven Convolutional Neural Network (CNN)  
    • Forward gravity model based on Granser’s method  
    • Synthetic and field data experiments  

    The goal is to provide fully reproducible results.

  tested_environment: |
    - Python 3.9 or 3.10
    - TensorFlow 2.x
    - Ubuntu / Windows 10 / macOS

  installation: |
    ### Create virtual environment
    python -m venv pinn_env

    ### Activate environment
    Linux/macOS:
      source pinn_env/bin/activate
    Windows:
      pinn_env\Scripts\activate.bat

    ### Install dependencies
    pip install -r requirements.txt

  repository_structure: |
    PINN/
        main.py
        model.py
        physics.py
        data_loader.py
        utils.py
        config.py

    CNN/
        main.py
        model.py
        data_loader.py
        utils.py
        config.py

    data/
        synthetic/
        field/

    results/
        figures/
        models/

  data_information: |
    ### Synthetic data
    Synthetic gravity anomalies and basement depths are generated using Granser’s forward model.

    Download dataset (Zenodo link in paper) and place inside:

    data/synthetic/

    ### Field data (optional)
    Place field data in:

    data/field/

    Supported formats:
    - CSV: x, y, gravity_anomaly
    - Numpy .npy arrays

  quick_start: |
    ### Run Physics-Informed Neural Network (PINN)

    cd PINN
    python main.py

    ### Run Convolutional Neural Network (CNN)

    cd CNN
    python main.py

  reproduction_steps: |
    ### Step 1 – Generate synthetic data
    python data/generate_synthetic.py

    ### Step 2 – Train PINN
    cd PINN
    python main.py --config config.py

    ### Step 3 – Train CNN
    cd CNN
    python main.py --config config.py

    ### Step 4 – Evaluate and compare
    python evaluate.py

    Results saved in:
    results/models/
    results/figures/

  outputs: |
    The example scripts will:

    ✔ train PINN and CNN models  
    ✔ predict basement depth  
    ✔ compare predicted vs true depth  
    ✔ compute metrics (MSE, MAE, R²)  
    ✔ generate plots  

  code_commenting_policy: |
    All Python files include English comments explaining:

    - loss function terms
    - physics-informed residuals
    - applied boundary conditions
    - data normalization
    - neural network architectures

  license: |
    MIT License. See LICENSE file.

  citation: |
    If you use this code, please cite the associated paper.

  contact: |
    For questions contact:
    Your Name
    Your Institution
    Your Email
