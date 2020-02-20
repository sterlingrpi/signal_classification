import numpy as np

Fs = 1500
f = 100
sample = 100000
f2 = [99, 100, 101]
period = 100

def import_1D_data(noise_level):
    print('generating 1D Data')
    x = np.arange(sample)
    signal_1 = np.sin(2*np.pi*f*x/Fs)
    noise = np.random.rand(sample)

    x_period = np.arange(period)
    signal_2 = []
    signal_2 = np.array(signal_2)
    while len(signal_2) < sample:
        for i in range(len(f2)):
            signal_2 = np.concatenate((signal_2, np.sin(2*np.pi*f2[i]*x_period/Fs)), axis=None)

    signal_2 = signal_2[0:sample]

    signal_1 = signal_1 + noise*noise_level
    signal_2 = signal_2 + noise*noise_level
    class_names = ['signal_1', 'signal_2']

    data = [signal_1, signal_2]
    data_shape = np.shape(signal_1)
    data = np.expand_dims(data, axis=2)

    print('done generating 1D data')

    return data, data_shape, class_names
