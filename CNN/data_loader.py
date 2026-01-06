import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    depth=scipy.io.loadmat('gravity_depth_field_21.mat')['depth']
    gravity=scipy.io.loadmat('gravity_depth_field_21.mat')['gravity']
    depthN=depth/10000
    
    # data split
    X_train_main, X_test, y_train_main, y_True = train_test_split(gravity, depthN, test_size=0.3, random_state=155)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_main, y_train_main, test_size=0.3, random_state=155)
    
    X_test=np.resize(X_test,(X_test.shape[0],101,101))
    y_True=np.resize(y_True,(y_True.shape[0],101,101))
    X_valid=np.resize(X_valid,(X_valid.shape[0],101,101))
    y_valid=np.resize(y_valid,(y_valid.shape[0],101,101))
    y_train=np.resize(y_train,(y_train.shape[0],101,101))
    X_train=np.resize(X_train,(X_train.shape[0],101,101))
    
    return X_train, X_valid, X_test, y_train, y_valid, y_True
