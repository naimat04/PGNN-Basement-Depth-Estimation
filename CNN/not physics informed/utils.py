import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

def R2_score(actual, pred):
    # Ensure both tensors have the same shape by squeezing the last dimension if needed
    # actual = tf.squeeze(actual)  # Remove last dimension if it's 1
    pred = tf.squeeze(pred, axis=-1)     # Remove last dimension if it's 1

    # Calculate R2 score
    ssres = tf.keras.backend.sum(tf.keras.backend.square(actual - pred))
    sstot = tf.keras.backend.sum(tf.keras.backend.square(actual - tf.keras.backend.mean(actual)))
    return 1 - (ssres / sstot)

def get_callbacks():
    base_path= '.'
    model_name='test'
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
    model_checkpoint = ModelCheckpoint(base_path + model_name +'.h5',monitor='val_R2_score',save_best_only=True, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.5, monitor='val_loss', patience=20, min_lr=0.000001, verbose=1)
    csv_logger = CSVLogger(base_path + model_name +".csv", append=True)
    
    return [model_checkpoint, reduce_lr, csv_logger, early_stopping]