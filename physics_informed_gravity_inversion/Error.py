y_train = (depth_train, gravity_train)
y_valid = (depth_valid, gravity_valid)

model.compile(optimizer='adam', loss=PhysicsLoss(alpha=0.7))

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32)



