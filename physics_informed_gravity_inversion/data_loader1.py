def load_data():
    depth=scipy.io.loadmat('exp_data.mat')['depthExp']
    gravity=scipy.io.loadmat('exp_data.mat')['gravityExp']
    depthN = depth / 10000

    # Split input and output as before
    X_train_main, X_test, y_train_main, y_True = train_test_split(gravity, depthN, test_size=0.3, random_state=155)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_main, y_train_main, test_size=0.3, random_state=155)

    # Resize for model input (add channel dimension)
    X_test = np.resize(X_test, (X_test.shape[0], 101, 101, 1))
    X_valid = np.resize(X_valid, (X_valid.shape[0], 101, 101, 1))
    X_train = np.resize(X_train, (X_train.shape[0], 101, 101, 1))

    y_True = np.resize(y_True, (y_True.shape[0], 101, 101, 1))
    y_valid = np.resize(y_valid, (y_valid.shape[0], 101, 101, 1))
    y_train = np.resize(y_train, (y_train.shape[0], 101, 101, 1))

    # Return tuple labels: (depth, gravity)
    return X_train, X_valid, X_test, (y_train, X_train), (y_valid, X_valid), (y_True, X_test)

