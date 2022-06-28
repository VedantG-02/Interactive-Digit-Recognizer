import tensorflow as tf
from model import create_model
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

cnn = create_model()

# compiling the cnn
cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# getting the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# converting data into numpy array, and splitting
train_images = np.array(train_data.iloc[:36000, 1:])
train_labels = np.array(train_data.iloc[:36000, 0])
validation_images = np.array(train_data.iloc[36000:, 1:])
validation_labels = np.array(train_data.iloc[36000:, 0])
test_images = np.array(test_data)

num_train = train_images.shape[0]
num_val = validation_images.shape[0]
num_test = test_images.shape[0]


for image in range(train_images.shape[0]):
    for pixel in range(784):
        if train_images[image][pixel] != 0:
            train_images[image][pixel] = 1
        train_images[image][pixel] = int(abs(train_images[image][pixel]-1))

for image in range(validation_images.shape[0]):
    for pixel in range(784):
        if validation_images[image][pixel] != 0:
            validation_images[image][pixel] = 1
        validation_images[image][pixel] = int(abs(validation_images[image][pixel]-1))

for image in range(test_images.shape[0]):
    for pixel in range(784):
        if test_images[image][pixel] != 0:
            test_images[image][pixel] = 1
        test_images[image][pixel] = int(abs(test_images[image][pixel]-1))


# converting into (28, 28) matrices
train_images = train_images.reshape(num_train, 28, 28, 1)
validation_images = validation_images.reshape(num_val, 28, 28, 1)
test_images = test_images.reshape(num_test, 28, 28, 1)

# training the model
cnn.fit(
    train_images, train_labels, 
    epochs = 20,
)

validation_pred = cnn.predict(validation_images)
validation_res = []

for pred in validation_pred:
    validation_res.append(np.argmax(pred))
validation_res = np.array(validation_res)

print("validation score", accuracy_score(validation_labels, validation_res), "\n")
print(confusion_matrix(validation_labels, validation_res), "\n")
print(classification_report(validation_labels, validation_res))
# for i in range(30):
#     print(validation_labels[i], validation_res[i])


# predicting test_images
test_pred = cnn.predict(test_images)


# for i in range(10):
#     img = tf.keras.preprocessing.image.array_to_img(train_images[i])
#     img.show()
#     print(train_labels[i])
#     # to check image and label
#     ans = 'n'
#     ans = input()
#     if(ans=='y'):
#         continue

# for i in range(10):
#     img = tf.keras.preprocessing.image.array_to_img(test_images[i])
#     img.show()
#     print(np.argmax(test_pred[i]))
#     # to check image and predicted label
#     ans = 'n'
#     ans = input()
#     if(ans=='y'):
#         continue

# SAVE MODEL
# ANN
# # saving the model in .json file
# model_json = cnn.to_json()
# with open('my_model_ann.json', 'w') as file:
#     file.write(model_json)

# # saving weights in .h5 file
# cnn.save_weights('my_model_ann.h5')

# CNN
# saving the model in .json file
model_json = cnn.to_json()
with open('my_model_cnn.json', 'w') as file:
    file.write(model_json)

# saving weights in .h5 file
cnn.save_weights('my_model_cnn.h5')

