import matplotlib.pyplot as plt
import numpy as np
import random
from models import get_model
from signal_batch_generator import get_data_sample, data_generator
from scipy.fftpack import fft

def visualy_inspect_generated_data(data, class_names, data_length):
    plt.figure(figsize=(10, 10))
    plt.suptitle('Training Images with Training Labels', fontsize=16)
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        class_number = random.randint(0, len(data) - 1)
        data_sample = get_data_sample(data, data_length, class_number)
        plt.plot(data_sample)
        plt.xlabel(class_names[class_number])
    plt.show()

def visualy_inspect_result(data, class_names, data_length, INPUT_CHANNELS, NUMBER_OF_CLASSES, loss_name):
    model = get_model(data_length, INPUT_CHANNELS, NUMBER_OF_CLASSES)
    model.load_weights('model_weights_' + loss_name + '.h5')
    batch_size = 12
    test_data, test_labels = data_generator(data, data_length, batch_size)
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print("test loss = ", test_loss)
    print("accuracy = ", test_acc)

    predictions = model.predict(test_data)

    num_rows = 4
    num_cols = 3
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    plt.suptitle('Test Data with Results', fontsize=16)
    for i in range(len(test_data)):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_data, class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.show()

def plot_image(i, predictions_array, true_labels, test_data, class_names):
    predictions_array, true_label, data = predictions_array[i], true_labels[i], test_data[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

#    plt.imshow(data, cmap=plt.cm.binary)
    plt.plot(data)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def plot_fft(y, Fs):
    T = 1 / Fs
    y = y[:, 0]
    N = len(y)
    yf = fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.grid()
    plt.show()
