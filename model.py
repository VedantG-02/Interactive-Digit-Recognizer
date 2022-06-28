import tensorflow as tf
import numpy as np

def create_model():
    # ANN
    # building the model
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     tf.keras.layers.Dense(1024, activation='tanh'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])

    # CNN
    # building the model
    model = tf.keras.models.Sequential([
         # first convolutional block
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='SAME', input_shape=[28, 28, 1]),
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        # second convolutional layer
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='SAME'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        # flattening the layer
        tf.keras.layers.Flatten(),
        # full connection
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # adding the output layer
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
   

    return model