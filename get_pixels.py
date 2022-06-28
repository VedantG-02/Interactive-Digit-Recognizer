import numpy as np
import cv2

# following function to get appropriate pixel values of (400, 400) canvas converted to (28, 28) size
def create_pixels(canvas):
    canvas_resized = cv2.resize(canvas, (28, 28))

    for row in range(canvas_resized.shape[0]):
        for col in range(canvas_resized.shape[1]):
            if canvas_resized[row][col] >= 0.5:
                canvas_resized[row][col] = 1
            else:
                canvas_resized[row][col] = 0

    return canvas_resized
