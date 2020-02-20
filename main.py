from plot_stuff import visualy_inspect_generated_data
from plot_stuff import visualy_inspect_result
from models import get_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from Generate_1D_Data import import_1D_data
from open_file_data import import_file_data
from signal_batch_generator import batch_generator
# Parameters
INPUT_CHANNELS = 1
NUMBER_OF_CLASSES = 2
epochs = 10
sample_per_epoch = 100
patience = epochs
batch_size = 100
data_length = 300
noise_level = 1
loss_name = "binary_crossentropy"

def train():
    model = get_model(data_length, INPUT_CHANNELS, NUMBER_OF_CLASSES)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint('model_weights_' + loss_name + '.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit_generator(
        generator=batch_generator(data, data_length, batch_size),
        epochs=epochs,
        steps_per_epoch=sample_per_epoch,
        callbacks=callbacks,
        validation_steps=1,
        validation_data=batch_generator(data, data_length, batch_size),
        )

    return model, history

if __name__ == '__main__':
    data, data_shape, class_names= import_1D_data(noise_level)
    #data, data_shape, class_names = import_file_data()
    visualy_inspect_generated_data(data, class_names, data_length)
    train()
    visualy_inspect_result(data, class_names, data_length, INPUT_CHANNELS, NUMBER_OF_CLASSES, loss_name)
