import tensorflow as tf
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import backend as K

img_width, img_height = 150, 150

print_files = True

train_data_dir = 'train'
validation_data_dir = 'validate'
batch_size = 1

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def test():
    model = load_model("model_cnn.h5")
    test_datagen = ImageDataGenerator(
        rescale=1. / 255)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    loss, acc = model.evaluate_generator(validation_generator)
    print('loss: {}, accuracy: {}'.format(loss, acc))

    classes = validation_generator.class_indices
    print('Classes: {}'.format(classes))
    classes_inv = {v: k for k, v in classes.items()}
    print('Inverted classes: {}'.format(classes_inv))

    """
    predictions = model.predict_generator(validation_generator, steps=10)
    for pred in predictions:
        max_pred = np.argmax(pred)
        print('Prediction: {} \n-> {}'.format(pred, classes_inv[max_pred]))"""
    i = 0
    max_test = validation_generator.__len__()
    num_errors = 0
    print('Checking {} non-shuffled images..'.format(max_test))
    # shuffling breaks filenames since they are not shuffled
    for batch_x, batch_y in validation_generator:
        if i >= max_test:
            break
        i += 1
        predicted = model.predict(batch_x, batch_size=batch_size)
        predicted = np.round(predicted, decimals=3)
        max_pred = np.max(predicted)
        predicted_class = classes_inv[np.argmax(predicted)]
        correct_class = classes_inv[np.argmax(batch_y)]
        try:
            filename = validation_generator.filenames[validation_generator.batch_index-1]
        except:
            filename = "error retrieving filename"
        if predicted_class != correct_class:
            num_errors += 1
            if print_files:
                print('Predicted wrong class ({:.3f} - {}): {} [{}]'.format(max_pred, predicted, predicted_class, filename))
    print('Errors in checked images: {}'.format(num_errors))




if __name__ == "__main__":
    test()
    K.clear_session()
