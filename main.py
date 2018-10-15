import cv2 as cv
import numpy as np
import threading
from Yolo.Yolov3 import Yolov3
from MobileNet.MobileNet import MobileNet

""" This projects aims to do image recognition on dual streams in real time.
    The end goal of this project is to obtain 3D information of classified objects.
"""
PI_ONE_URL = 'http://192.168.1.251:8000/stream.mjpg'
PI_TWO_URL = 'http://192.168.1.252:8000/stream.mjpg'

left_net = None
right_net = None

class CamThread(threading.Thread):
    def __init__(self, previewName, camStream, net):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camStream = camStream
        self.net = net

    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camStream, self.net)


def camPreview(previewName, camStream, net):
    cv.namedWindow(previewName)
    cam = cv.VideoCapture(camStream)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        net.setFrame(frame)
        net.draw_predictions(frame)
        cv.imshow(previewName, frame)
        rval, frame = cam.read()
        if cv.waitKey(5) & 0xFF == 27:  # Exit on ESC
            break
    cv.destroyWindow(previewName)
    cam.release()


def main():
    # Create two threads as follows
    left_net = MobileNet()  # Can change to Yolov3 if wanted
    right_net = MobileNet()
    left_net.start()
    right_net.start()

    thread_left = CamThread("Camera Left", PI_ONE_URL, left_net)
    thread_right = CamThread("Camera Right", PI_TWO_URL, right_net)

    thread_left.start()
    thread_right.start()



if __name__ == '__main__':
    main()
