y_train = (depth_train, gravity_train)
y_valid = (depth_valid, gravity_valid)

model.compile(optimizer='adam', loss=PhysicsLoss(alpha=0.7))

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32)

X_train = X_train[..., np.newaxis]  # shape (N,101,101,1)
depth_train = depth_train[..., np.newaxis]
gravity_train = gravity_train[..., np.newaxis]

