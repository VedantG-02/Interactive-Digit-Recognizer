import tensorflow as tf
from get_pixels import create_pixels
from draw import create_canvas
import cv2
import numpy as np

# getting the canvas
canvas = create_canvas()
test_digit = create_pixels(canvas)

# # load ann model
# model = tf.keras.models.model_from_json(open("my_model_ann.json", "r").read())
# model.load_weights('my_model_ann.h5')

# load cnn model
model = tf.keras.models.model_from_json(open("my_model_cnn.json", "r").read())
model.load_weights('my_model_cnn.h5')

if __name__ == '__main__':
    # prediction of digit
    num = np.argmax(model.predict(test_digit.reshape(1, 28, 28)))
    text = "Prediction is " + str(num)
    # prediction window
    img = np.full((100, 250), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, text, (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Prediction', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
