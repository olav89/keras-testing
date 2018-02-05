"""

https://github.com/jacobgil/keras-grad-cam
With some minor tweaks to get things working on new Keras version

"""

from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2

num_classes = 3
model_path = "model_cnn.h5"
test_image = "test.jpg"

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image():
    img = image.load_img(test_image, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x) # dont know what VGG16 does in preprocess, readd if needed
    return x

def grad_cam(input_model, image, category_index, layer_name):

    model = input_model
    #print(model.summary())
    target_layer = lambda x: target_category_loss(x, category_index, num_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))


    loss = K.sum(model.layers[-1].output)
    #conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    test = [l for l in model.layers if l.name.find(layer_name) > -1]
    print(test)
    conv_output = test[0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (150, 150))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

preprocessed_input = load_image()

model = keras.models.load_model(model_path)

predictions = model.predict(preprocessed_input)
print(predictions)

predicted_class = np.argmax(predictions)
cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, 'b4_c2')
cv2.imwrite("gradcam.jpg", cam)

K.clear_session()
