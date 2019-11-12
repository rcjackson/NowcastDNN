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


radar_grids_path = '/home/rjackson/bebop_earthscience/rjackson/deep_learning/2006/'
tfrecords_path = '/home/rjackson/tfrecords/2006/2frames_future/'
num_frames_in_past = 3
my_shape = (201, 201)
is_training = True
shuffle = False

def create_tf_record(radar_list, scan_no, num_previous_frames=3, num_frames_future=2):
    # Get radar file from bebop
    if scan_no < num_previous_frames:
        print("Not enough frames in the past.")
        return 
    elif scan_no+num_frames_future > len(radar_list):
        print("Not enough frames in future!")
        return
    
    # First get previous frames
    # Normalization: -20 dBZ = 0, 60 dBZ = 1 
    Znorm_list = []    
    times = np.zeros(num_previous_frames, dtype=datetime)
    for i in range(num_previous_frames-1, -1, -1):
        grid = xr.open_dataset(radar_list[scan_no-i])
        try:
            Zn = grid.Znorm.fillna(-20).values
            if (np.nanmax(Zn) > 0):
                Zn = (Zn + 20.)/(80)
            else:
                Zn = np.zeros_like(Zn)
            Znorm_list.append(Zn)
        except:
            return
        times[i] = grid.time.values
        grid.close()
    Znorm = np.stack(Znorm_list)
    dt = np.diff(times)
    #if(dt.max() > timedelta(minutes=60)):
    #    print("Data is not continuous enough to put into training set!")
    #    return

    dt = np.concatenate([np.array([0]), dt])
    # Then get the future frame 
    try:
        grid = xr.open_dataset(radar_list[scan_no+num_frames_future])
        Znorm_future = grid.Znorm.fillna(-20).values
        max_ref = np.nanmax(Znorm_future)
        min_ref = np.nanmin(Znorm_future)
        if(max_ref > 0):
            Znorm_future = (Znorm_future + 20)/80.
        else:
            Znorm_future = np.zeros_like(Znorm_future)
        Znorm_future[~np.isfinite(Znorm_future)] = 0
    except:
        return
    print("Past times:")
    print(times)
    print("Future time: ")
    print(grid.time.values)
    grid_time = grid.time.values
    grid.close()
    norm = Znorm/np.nanmax(Znorm)
    shp = norm.shape
    my_shape = shp
    width = shp[0]
    height = shp[1]
    dt_future = grid_time - times[-1]
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
                        'image_raw': _bytes_feature(Znorm),
                        'label': _bytes_feature(Znorm_future),
                        'num_frames_in_past': _int64_feature(num_previous_frames),
                        'num_frames_in_future': _int64_feature(num_frames_future),
                        'dt_past': _bytes_feature(dt),
                        'dt_future': _float_feature(dt_future),
                        'max_ref': _float_feature(max_ref)
                    }))
    writer.write(example.SerializeToString())

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
