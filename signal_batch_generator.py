import random
import numpy as np

def get_data_sample(data, data_length, class_number):
    start = random.randint(0, np.shape(data)[1] - data_length)
    end = start + data_length
    data_slice = data[class_number][start:end]

    return data_slice

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