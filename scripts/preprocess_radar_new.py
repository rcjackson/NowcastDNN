import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from tensorflow.keras.layers import LSTM, ConvLSTM2D, Conv2D, ZeroPadding2D
import numpy as np
import xarray as xr
from glob import glob
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")


radar_grids_path = '/home/rjackson/bebop_earthscience/rjackson/deep_learning/2005/'
tfrecords_path = '/home/rjackson/tfrecords/2005/'
num_frames_in_past = 3
my_shape = (201, 201)
is_training = True
shuffle = False

def create_tf_record(radar_list, scan_no):
    # Get radar file from bebop
    
    
    # First get previous frames
    # Normalization: -20 dBZ = 0, 60 dBZ = 1 
    try:
        grid = xr.open_dataset(radar_list[scan_no])
        Zn = grid.Znorm.fillna(-20).values
        if (np.nanmax(Zn) > 0):
            Zn = (Zn + 20.)/(80)
        else:
            Zn = np.zeros_like(Zn)
    except:
        return
    times = grid.time.values
    grid.close()
    shp = Zn.shape
    my_shape = shp
    width = shp[0]
    height = shp[1]
    
    #if dt_future > timedelta(minutes=60):
    #    print("Data not continous")
    #    return

    fname = tfrecords_path + radar_list[scan_no][-16:] + '.tfrecord'
    writer = tf.python_io.TFRecordWriter(fname)
    
    #norm = norm.SerializeToString()
    example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'width': _int64_feature(width),
                        'height': _int64_feature(height),
                        'image_raw': _bytes_feature(Zn),
                        'time': _float_feature(times),
                    }))
    writer.write(example.SerializeToString())
    print(times)
    del writer

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
    file_list = sorted(glob(radar_grids_path + '/**/*.cdf', recursive=True))
    print("About to process %d files" % len(file_list))
    for i in range(len(file_list)):
        create_tf_record(file_list, i)
