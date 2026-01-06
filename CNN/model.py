import tensorflow as tf
from tensorflow.keras import layers, models

def build_xception_model(input_shape=(101, 101, 1)):
    input_tensor = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding='same', name='Conv1')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    skip1 = x

    x = layers.Conv2D(64, (3, 3), use_bias=False, padding='same', name='Conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    skip2 = x

    residual = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='res1')(x)
    residual = layers.BatchNormalization()(residual)
    skip3 = residual

    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='Conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='Conv4')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        x = layers.add([x, residual])

        residual = layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='deConv1')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='deConv2')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='deConv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Upsampling to match the input size
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(128, (2, 2), padding='same', use_bias=False, name='deConv4')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Concatenate()([x, skip3])

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, (2, 2), padding='valid', use_bias=False, name='deConv5')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Concatenate()([x, skip2])

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(32, (2, 2), padding='valid', use_bias=False, name='deConv6')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Output layer (regression)
    output_tensor = layers.Conv2D(1, (1, 1), activation='linear', use_bias=False, name='deConv7')(x)

    # Define model
    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    
    # Compile the model with standard loss functions
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mse'])

    return model