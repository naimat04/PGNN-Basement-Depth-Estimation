from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation, Conv2D, SeparableConv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import backend as K
import scipy.io
import scipy.ndimage.filters as fil
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, Model
from scipy.fft import fft2, ifft2
import math

from model import build_xception_physics, PhysicsLoss
from data_loader import load_data
from utils import R2_score, get_callbacks

# Example usage
if __name__ == "__main__":
    # Build model
    model = build_xception_physics()

    # Compile with physics-informed loss
    physics_loss = PhysicsLoss(X_obs=X_train, alpha=0.7)  # 70% data, 30% physics
    model.compile(optimizer='adam', loss=physics_loss, metrics=[R2_score])

    # Summary
    model.summary()

    # Load data
    X_train, X_valid, X_test, y_train, y_valid, y_True = load_data()

    # Recompile for training
    # model.compile(optimizer='adam', loss='mse', metrics=[R2_score])

    # Get callbacks
    callbacks = get_callbacks()

    # Train model
    history = model.fit(X_train, y_train,
              validation_data=[X_valid,y_valid],
                      epochs=100, batch_size=32,
                       callbacks=callbacks, verbose=1)

    # Save weights
    base_path = '.'
    model_name = 'test'
    model.save_weights(base_path + model_name + '.weights.h5')
