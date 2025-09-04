import tensorflow as tf
from tensorflow.keras import layers, models
from physics import calculate_gravity_field

class PhysicsLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, name="physics_loss"):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        depth_true, gravity_obs = y_true  # unpack tuple

        # Data loss between true and predicted depth
        data_loss = tf.reduce_mean(tf.square(depth_true - y_pred))

        # Physics loss between gravity computed from predicted depth and observed gravity
        calculated_gravity = calculate_gravity_field(y_pred)
        physics_loss = tf.reduce_mean(tf.square(calculated_gravity - gravity_obs))

        return self.alpha * data_loss + (1 - self.alpha) * physics_loss

def build_xception_physics(input_shape=(101, 101, 1)):
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

    output_tensor = layers.Conv2D(1, (1, 1), activation='linear', use_bias=False, name='deConv7')(x)

    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    
    # Do NOT compile here; compile outside to allow custom loss function
    return model
