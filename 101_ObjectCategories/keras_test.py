import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import backend as K

img_width, img_height = 150, 150

train_data_dir = 'train'
validation_data_dir = 'validate'
num_classes = 3
epochs = 5
batch_size = 8

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
        class_mode='categorical')

    loss, acc = model.evaluate_generator(validation_generator)
    print('loss: {}, accuracy: {}'.format(loss, acc))


if __name__ == "__main__":
    test()
