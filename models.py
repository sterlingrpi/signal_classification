from keras.layers import Input, MaxPooling2D, UpSampling2D, Conv1D, Conv2D, Reshape, Conv2DTranspose, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from keras.layers.core import Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.models import Model

def get_model(data_length, INPUT_CHANNELS, NUMBER_OF_CLASSES):
    inputs = Input((data_length, INPUT_CHANNELS))

    model = get_1D_DNN_sequential(NUMBER_OF_CLASSES, data_length)
#    model = get_1D_DNN(inputs, NUMBER_OF_CLASSES)
#    model = get_DNN_Sequential(inputs, NUMBER_OF_CLASSES, image_shape)
#    model = get_DNN(inputs, NUMBER_OF_CLASSES, image_shape)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    return model

def get_1D_DNN_sequential(NUMBER_OF_CLASSES, data_length):
    model = Sequential()
    #model.add(Reshape((data_shape, 1), input_shape=(data_shape)))
    model.add(Conv1D(filters=100, kernel_size=10, activation='relu', input_shape=(data_length, 1)))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))

    return model

def get_1D_DNN(inputs, NUMBER_OF_CLASSES):
    x = BatchNormalization()(inputs)

    x = Conv1D(100, 10, activation='relu')(x)
    x = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)

    act = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=act)

    return model

def get_DNN(inputs, n_classes, image_shape):
    x = BatchNormalization()(inputs)

    x = Conv2D(32, (3, 3), activation='relu', padding='valid', name='block1_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), name='block1_pool')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='valid', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), name='block2_pool')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='valid', name='block3_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), name='block3_pool')(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(100)(x)
    x = Activation('relu')(x)
    x = Dense(2)(x)
#    x = Activation('softmax')(x)

    act = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=act)

    return model

def get_DNN_Sequential(inputs, n_classes, image_shape):
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape))
    model.add(Activation('relu'))  # tanh
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))  # tanh
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))  # tanh
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model
