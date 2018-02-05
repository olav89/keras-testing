import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D
from keras import backend as K
from split_data import split_files
import os
import shutil
from keras.utils import plot_model

model_name = "model_cnn.h5"
img_width, img_height = 150, 150

train_data_dir = 'train'
validation_data_dir = 'validate'
log_dir = 'logs'
num_classes = 2
nb_train_samples, nb_validation_samples = split_files(num_classes=num_classes, split_ratio=0.2)
epochs = 5
batch_size = 32

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
    # remove old logs
    shutil.rmtree('{}/{}'.format(os.getcwd(), log_dir), ignore_errors=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3,
                                                 verbose=0, mode='auto'),
            keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss', verbose=0, save_best_only=True,
                                                   save_weights_only=False, mode='auto', period=1),
            keras.callbacks.TensorBoard(log_dir='./{}'.format(log_dir), histogram_freq=0, batch_size=batch_size, write_graph=True,
                                               write_grads=False, write_images=False, embeddings_freq=0,
                                               embeddings_layer_names=None, embeddings_metadata=None),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=0, mode='auto',
                                              epsilon=0.0001, cooldown=0, min_lr=1e6)])
    print("Training completed.")


def make_model_cnn():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape, name="b1_c1"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name="b1_c2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name="b2_c1"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name="b2_c2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name="b3_c1"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name="b3_c2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name="b4_c1"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name="b4_c2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
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
    K.clear_session()  # attempt at fixing py3 None Type bug
