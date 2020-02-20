import numpy as np

def open_radar_file(filename, dtype, count):
    f = open(filename)
    y = np.fromfile(file=f, dtype=dtype, count=count)
    f.close()
    return y

def import_file_data():
    filename = "C/data/signal_1_data_file"
    filename2 = "C/data/signal_2_data_file"
    count = 1000000
    dtype = 'float32'

    signal_1 = open_radar_file(filename, dtype, count)
    singal_2 = open_radar_file(filename2, dtype, count)

    data = [signal_1, singal_2]
    data_shape = np.shape(signal_1)
    data = np.expand_dims(data, axis=2)

    class_names = ['signal_1', 'singal_2']

    return data, data_shape, class_names
