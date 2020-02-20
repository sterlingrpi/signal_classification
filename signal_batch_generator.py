import random
import numpy as np

def get_data_sample(data, data_length, class_number):
    start = random.randint(0, np.shape(data)[1] - data_length)
    end = start + data_length
    data_sample = data[class_number][start:end]
    #data_sample = pitch_shift(data_sample, shift=0.1)
    data_sample = amplitude_adjust(data_sample, amplitude=0.5)
    return data_sample

def pitch_shift(data_sample, shift):
    data_sample_mean = np.mean(data_sample)
    data_sample -= data_sample_mean #remove DC offset
    yf = np.fft.fft(data_sample)
    len_yf = len(yf)
    shift = int(shift*len_yf*(random.random() - 0.5))
    yf = np.roll(yf, shift)
    data_sample = np.fft.irfft(yf, axis=0)
    data_sample += data_sample_mean #add DC offset back
    return data_sample

def amplitude_adjust(data_sample, amplitude):
    data_sample = data_sample*(random.uniform(1-amplitude, 1+amplitude))
    return data_sample

def batch_generator(data, data_length, batch_size):
    while True:
        data_list = []
        label_list = []
        for i in range(batch_size):
            class_number = random.randint(0, len(data) - 1)
            data_sample = get_data_sample(data, data_length, class_number)
            data_list.append(data_sample)
            label_list.append(class_number)

        data_list = np.array(data_list, dtype=np.float32)
        label_list = np.array(label_list, dtype=np.float32)

        yield data_list, label_list

def data_generator(data, data_length, batch_size):
    data_list = []
    label_list = []
    for i in range(batch_size):
        class_number = random.randint(0, len(data) - 1)
        data_sample = get_data_sample(data, data_length, class_number)
        data_list.append(data_sample)
        label_list.append(class_number)

    data_list = np.array(data_list, dtype=np.float32)
    label_list = np.array(label_list, dtype=np.int)

    return data_list, label_list