import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    depth=scipy.io.loadmat('exp_data.mat')['depthExp']
    gravity=scipy.io.loadmat('exp_data.mat')['gravityExp']
    depthN = depth / 10000

    # Split
    X_train, X_test, y_train, y_True = train_test_split(gravity, depthN, test_size=0.3, random_state=155)
    X_valid, X_test, y_valid, y_True = train_test_split(X_test, y_True, test_size=0.3, random_state=155)

    # Reshape
    def reshape(data):
        return np.expand_dims(np.resize(data, (data.shape[0], 101, 101)), axis=-1)

    gravity_train = reshape(X_train)
    gravity_valid = reshape(X_valid)
    gravity_test = reshape(X_test)

    depth_train = reshape(y_train)
    depth_valid = reshape(y_valid)
    depth_true = reshape(y_True)

    y_train = np.concatenate([depth_train, gravity_train], axis=-1)
    y_valid = np.concatenate([depth_valid, gravity_valid], axis=-1)
    y_true  = np.concatenate([depth_true,  gravity_test], axis=-1)
    return gravity_train, gravity_valid, gravity_test, y_train, y_valid, y_true
