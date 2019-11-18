import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from tensorflow.keras.layers import LSTM, ConvLSTM2D, Conv2D, ZeroPadding2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import pyart
import numpy as np
import xarray as xr
from glob import glob
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")


tfrecords_path = '/home/rjackson/tfrecords/2006/*'
num_frames_in_past = 3
num_frames_in_future = 1
my_shape = (201, 201)
is_training = True
shuffle = False

def input_fn():
    def parse_record(record):
        feature={'width': tf.FixedLenFeature([], tf.int64, default_value=0),
                 'height': tf.FixedLenFeature([], tf.int64, default_value=0),
                 'image_raw': tf.FixedLenFeature([], tf.string, default_value=""),
                 'num_frames_in_past': tf.FixedLenFeature([], tf.int64, default_value=0),
                 'num_frames_in_future': tf.FixedLenFeature([], tf.int64, default_value=0),
                 'dt_past': tf.FixedLenFeature([], tf.string, default_value=""),
                 'time': tf.FixedLenFeature([], tf.string, default_value=""),
                 'dt_future': tf.FixedLenFeature([], tf.float32, default_value=0.),
                 }
        features = tf.io.parse_single_example(record, feature)
        image_shape = (num_frames_in_past, my_shape[0], my_shape[1], 1) 
        features['image_raw'] = tf.decode_raw(features['image_raw'], tf.float32)
        features['image_raw'] = tf.reshape(features['image_raw'], shape=list(image_shape))
             
        return {'image_raw': features['image_raw']}

    def 
    dataset = tf.data.TFRecordDataset(file_list)

    if is_training:
        if shuffle:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
                      buffer_size=shuffle_buffer, seed=seed))
        else:
            dataset = dataset.repeat()

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                parse_record,
                batch_size=30,
                num_parallel_batches=4,
                drop_remainder=True))
    
    return dataset

def _int64_feature(value):
    """Creates a tf.Train.Feature from an int64 value."""
    if value is None:
        value = []
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Creates a tf.Train.Feature from a bytes value."""
    if value is None:
        value = []
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)
        value = bytes(value)
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Creates a tf.Train.Feature from a bytes value."""
    if value is None:
        value = []
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)
        value = bytes(value)
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

if __name__ == "__main__":
    hidden_size = 5
    use_dropout=True
    num_steps = 3
    the_shape = my_shape
    def model_function():
        new_shape = (num_steps, the_shape[0], the_shape[1], 1)
        #new_shape_conv10 = (num_steps, the_shape[0]-conv_window, the_shape[1]-conv_window, 1)
    
        model = Sequential()
        model.add(ConvLSTM2D(filters=1, kernel_size=(3, 3),
            input_shape=new_shape, data_format='channels_last', return_sequences=True,
            padding='same', activation='relu'))
        model.add(ConvLSTM2D(filters=1, kernel_size=(9, 9),
            data_format='channels_last', return_sequences=True,
            padding='same', activation='relu'))
        model.add(ConvLSTM2D(filters=1, kernel_size=(9, 9),
            data_format='channels_last', return_sequences=True,
            padding='same', activation='relu'))
        model.add(ConvLSTM2D(filters=1, kernel_size=(3, 3),
            data_format='channels_last', return_sequences=False,
            padding='same', activation='relu'))
        model.add(Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                     data_format='channels_last', activation='sigmoid'))
        #odel.add(ConvLSTM2D(1, 10, input_shape=new_shape_conv10, data_format='channels_last', return_sequences=True))
        #if use_dropout:
        #    model.add(Dropout(0.5))
        optimizer = SGD(clipnorm=1.0, clipvalue=0.5) 
        model.compile(loss='mean_squared_error', optimizer=optimizer)
    
        return model
    
    mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with tf.Session() as sess:
        my_model = model_function()
        my_model.summary()
        dataset = input_fn()
        checkpointer = ModelCheckpoint(filepath='/home/rjackson/DNNmodel/model-{epoch:02d}.hdf5', verbose=1)
        my_model.fit(dataset, None, epochs=30, callbacks=[checkpointer])
        
        
