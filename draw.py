import cv2
import numpy as np
import random

drawing = False
x=-1
y=-1
count=1

def create_canvas():
    # creating white canvas
    canvas = np.full((400, 440), 255, dtype=np.uint8)
    width, height = canvas.shape
    #print(canvas)
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas = cv2.putText(canvas, 'Press q/x to quit window', (20, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    # cv2.line(canvas, (0, 0), (0, width), (0, 0, 0), 3)
    # cv2.line(canvas, (height, 0), (height, width), (0, 0, 0), 3)
    # cv2.line(canvas, (0, 0), (height, 0), (0, 0, 0), 2)
    # cv2.line(canvas, (0, width), (height, width), (0, 0, 0), 3)

    def draw(event, current_x, current_y, flags, param):
        global x, y, drawing, count
        # when left mouse button is clicked on the window
        if event == cv2.EVENT_LBUTTONDOWN:
            if count==1:
                drawing = True
                x = current_x
                y = current_y
        # when mouse is moved, given left button is still down
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.line(canvas, (x, y), (current_x, current_y), (0, 0, 0), 25)
                x = current_x
                y = current_y
        # when finger is picked from the left button
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False # stops taking response further
            if count==1:
                cv2.line(canvas, (x, y), (current_x, current_y), (0, 0, 0), 25)
            count+=1


    cv2.namedWindow('canvas')
    cv2.setMouseCallback('canvas', draw)

    while(1):
        cv2.imshow('canvas', canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == ord('x'):
            break 

    #print(canvas)
    return canvas[:, 40:]
