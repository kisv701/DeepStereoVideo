import numpy as np
from threading import Thread
import time
import cv2


class MobileNet(Thread):
    def __init__(self, frame=np.zeros((416, 416, 3), dtype=np.uint8)):
        Thread.__init__(self)
        # Load the Caffe model
        self.net = cv2.dnn.readNetFromCaffe('MobileNet/MobileNetSSD_300x300.prototxt',
                                            'MobileNet/MobileNetSSD_train.caffemodel')
        self.scale = 0.007843
        self.input_width = 300
        self.input_height = 300
        self.detections = None
        self.thresh = 0.2

        self.predicting = False
        self.drawing = False
        self.frame = frame

        # Labels of network.
        self.classes = {0: 'background',
                        1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                        5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                        10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                        14: 'motorbike', 15: 'person', 16: 'pottedplant',
                        17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))



    def setFrame(self, frame):
        self.frame = frame

    def predict(self, image):
        starttime = time.time()
        self.predicting = True

        frame_resized = cv2.resize(image, (self.input_width, self.input_height))
        blob = cv2.dnn.blobFromImage(frame_resized, self.scale, (300, 300), (127.5, 127.5, 127.5), False)
        self.net.setInput(blob)

        outs = self.net.forward()

        endtime = time.time()
        print('Predictions time {0:.2f} [s] ~ {1:.2f} [fps]'.format(endtime - starttime, 1/(endtime - starttime)))

        while self.drawing:
            pass
        self.detections = outs
        self.predicting = False

    def draw_predictions(self, image):
        self.drawing = True

        if self.detections is None:
            self.drawing = False
            return
        # For get the class and location of object detected,
        # There is a fix index for class, location and confidence
        # value in @detections array .
        for i in range(self.detections.shape[2]):
            confidence = self.detections[0, 0, i, 2]  # Confidence of prediction
            if confidence > self.thresh:  # Filter prediction
                class_id = int(self.detections[0, 0, i, 1])  # Class label
                # Object location
                xLeftBottom = int(self.detections[0, 0, i, 3] * self.input_width)
                yLeftBottom = int(self.detections[0, 0, i, 4] * self.input_height)
                xRightTop = int(self.detections[0, 0, i, 5] * self.input_width)
                yRightTop = int(self.detections[0, 0, i, 6] * self.input_height)

                # Factor for scale to original size of frame
                heightFactor = image.shape[0] / 300.0
                widthFactor = image.shape[1] / 300.0
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom)
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop = int(widthFactor * xRightTop)
                yRightTop = int(heightFactor * yRightTop)
                # Draw location of object
                cv2.rectangle(image, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))

                # Draw label and confidence of prediction in frame resized
                if class_id in self.classes:
                    label = "{0} : {1:.2f}".format(self.classes[class_id], confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(image, (xLeftBottom, yLeftBottom - labelSize[1]),
                                  (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                  (255, 255, 255), cv2.FILLED)
                    cv2.putText(image, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        self.drawing = False

    def run(self):
        print("Running..")
        while True:
            if not self.predicting:
                self.predict(self.frame)
