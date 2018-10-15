import cv2
import numpy as np
from Yolo.Yolov3 import Yolov3 as Yolov3
from MobileNet.MobileNet import MobileNet

PI_ONE_URL = 'http://192.168.1.251:8000/stream.mjpg'

def main():
    net = MobileNet()
    net.start()
    cap = cv2.VideoCapture(PI_ONE_URL)

    if cap.isOpened():  # try to get the first frame
        rval, frame = cap.read()
    else:
        rval = False

    while rval:

        net.draw_predictions(frame)
        cv2.imshow('Yolo', frame)
        rval, frame = cap.read()
        net.setFrame(frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC
            break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()