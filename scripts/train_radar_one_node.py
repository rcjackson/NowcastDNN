import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from tensorflow.keras.layers import LSTM, ConvLSTM2D, Conv2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.contrib.data.python.ops import sliding

import pyart
import numpy as np
import xarray as xr
from glob import glob
import matplotlib.pyplot as plt
import warnings
import sys
from datetime import datetime
warnings.filterwarnings("ignore")


tfrecords_path = '/home/rjackson/tfrecords/2006/*'
num_frames_in_past = 3
num_frames_in_future = int(sys.argv[1])
my_shape = (201, 201)
is_training = True
shuffle = False

def input_fn():
    def parse_record(record):
        feature={'width': tf.FixedLenFeature([], tf.int64, default_value=0),
                 'height': tf.FixedLenFeature([], tf.int64, default_value=0),
                 'image_raw': tf.FixedLenFeature([], tf.string, default_value=""),
                 'time': tf.FixedLenFeature([], tf.float32, default_value=0.),
                }
        features = tf.io.parse_single_example(record, feature)
        image_shape = (my_shape[0], my_shape[1], 1)
        features['image_raw'] = tf.decode_raw(features['image_raw'], tf.float32)
        features['image_raw'] = tf.reshape(features['image_raw'], shape=image_shape)
        return {'image_raw': features['image_raw']}

    def make_inputs(record):
        return {'conv_lst_m2d_input': record}

    def make_labels(record):
        return {'conv2d': record}

    file_list = sorted(glob(tfrecords_path))

    dataset = tf.data.TFRecordDataset(file_list)

    if is_training:
        if shuffle:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
                      buffer_size=shuffle_buffer, seed=seed))
        else:
            dataset = dataset.repeat()

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(parse_record)
    features = dataset.window(size=num_frames_in_past, shift=1)
    features = features.flat_map(lambda x: x["image_raw"].batch(num_frames_in_past)) 
    features = features.map(make_inputs)
    features = features.batch(30)
    labels = dataset.skip(num_frames_in_future)
    labels = labels.map(lambda x: x["image_raw"])
    labels = labels.map(make_labels)
    labels = labels.batch(30)
    dataset = tf.data.Dataset.zip((features, labels))
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
   
    the_shape = my_shape
    
    def model_function(num_channels=1):
        new_shape = (None, the_shape[0], the_shape[1], num_channels)
       
        #new_shape_conv10 = (num_steps, the_shape[0]-conv_window, the_shape[1]-conv_window, 1)
        kern_size = (5, 5)
        model = Sequential()
        
        model.add(ConvLSTM2D(filters=num_channels, kernel_size=kern_size,
                             data_format='channels_last', 
                             recurrent_activation='hard_sigmoid',
                             activation='tanh', padding='same',
                             return_sequences=True,
                             input_shape=new_shape))
               
                             
        model.add(BatchNormalization())
#        max_pool1 = MaxPooling3D(pool_size=(1, 2, 2))(batch_norm1)
        
        model.add(ConvLSTM2D(filters=num_channels, kernel_size=kern_size,
                             data_format='channels_last',
                             recurrent_activation='hard_sigmoid',
                             activation='tanh', padding='same',
                             return_sequences=True,
                             ))
               
        model.add(BatchNormalization())
		#max_pool2 = MaxPooling3D(pool_size=(1, 2, 2))(batch_norm2)

        model.add(ConvLSTM2D(filters=num_channels, kernel_size=kern_size,
                             data_format='channels_last',
                             recurrent_activation='hard_sigmoid',
                             activation='tanh', padding='same',
                             return_sequences=False))        
        model.add(BatchNormalization())
        #model.add(Dropout(0.3))
        #model.add(ConvLSTM2D(filters=num_channels, kernel_size=kern_size,
        #                     data_format='channels_last',
        #                     recurrent_activation='hard_sigmoid',
        #                     activation='tanh', padding='same',
        #                     return_sequences=False))
       
        model.add(BatchNormalization())

        
        model.add(Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                  data_format='channels_last', activation='sigmoid'))
        #odel.add(ConvLSTM2D(1, 10, input_shape=new_shape_conv10, data_format='channels_last', return_sequences=True))
        #if use_dropout:
        #    model.add(Dropout(0.5))
        #model = Model(inputs=my_input, outputs=output_layer, name='NowcastDNN') 
        model.compile(loss='mean_squared_error', optimizer='SGD')  
        return model
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with tf.Session() as sess:
        my_model = model_function()
        my_model.summary()
        dataset = input_fn()
        checkpointer = ModelCheckpoint(filepath=('/home/rjackson/DNNmodel/model-%dframes-{epoch:03d}.hdf5'
                                                 % num_frames_in_future), verbose=1)
        my_model.fit(dataset, None, epochs=300, callbacks=[checkpointer], steps_per_epoch=1600)
