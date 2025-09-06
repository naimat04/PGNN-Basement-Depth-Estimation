import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

def R2_score(actual, pred):
    # Take only depth channel
    actual_depth = actual[..., 0]
    pred_depth   = pred[..., 0]

    ssres = tf.keras.backend.sum(tf.keras.backend.square(actual_depth - pred_depth))
    sstot = tf.keras.backend.sum(tf.keras.backend.square(actual_depth - tf.keras.backend.mean(actual_depth)))
    return 1 - ssres/(sstot + tf.keras.backend.epsilon())


def get_callbacks():
    base_path= '.'
    model_name='test'
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
    model_checkpoint = ModelCheckpoint(base_path + model_name +'.h5',monitor='val_R2_score',save_best_only=True, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.5, monitor='val_loss', patience=20, min_lr=0.000001, verbose=1)
    csv_logger = CSVLogger(base_path + model_name +".csv", append=True)
    
    return [model_checkpoint, reduce_lr, csv_logger, early_stopping]
