import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from split_data import split_files
import os
from keras.utils import plot_model

img_width, img_height = 150, 150

train_data_dir = 'train'
validation_data_dir = 'validate'
num_classes = 3
nb_train_samples, nb_validation_samples = split_files(num_classes=num_classes, split_ratio=0.2)
epochs = 1
batch_size = 8

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def train():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(
        rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    model = make_model_cnn()
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save('model_cnn.h5')


def make_model_cnn():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    plot_model(model, to_file='model.png', show_shapes=True) # plot model to file

    return model


# Code-snippet from:
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
def save_preview_images(train_datagen):
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    img = load_img('train/accordion/image_0001.jpg')
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in train_datagen.flow(x, batch_size=1,
                                    save_to_dir=os.getcwd() + '/preview/',
                                    save_prefix='preview', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely


if __name__ == "__main__":
    train()