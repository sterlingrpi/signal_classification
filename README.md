This project learns and classifies/discerns different 1D signals using tensorflow like. The signals can either be generated synthetically or loaded from a file. The synthetically generated data creates two signals; one with a constant frequency and another one with a modulated frequency. Variable amounts of noise are added to the synthetic data. The amplitudes and frequency (pitch shift) are randomly added to the data during training as part of the data generator. This way it will learn to generalize modulation schemes of the signals and not the relative amplitudes and frequencies.

Below is an illustrative result after training. The signals are shown next to confidence bar graphs. Blue means it got the prediction correct. Red means it predicted the wrong class.

![Image description](https://github.com/sterlingrpi/signal_classification/blob/master/signal_classification_result.png)

To run the code, run main. You can uncomment the import_file_data() line in main to use your own data files. Within open_file_data.py you can define the pathes of your files.
