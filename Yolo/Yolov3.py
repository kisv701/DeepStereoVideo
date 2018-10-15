import cv2
import numpy as np
from threading import Thread
import time


class Yolov3(Thread):
    def __init__(self, frame=np.zeros((416, 416, 3), dtype=np.uint8)):
        Thread.__init__(self)
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.scale = 0.00392

        self.predicting = False
        self.drawing = False
        self.frame = frame

        with open('Yolo/yolov3_classes.txt', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.net = cv2.dnn.readNet('Yolo/yolov3.weights', 'Yolo/yolov3.cfg')

    def setFrame(self, frame):
        self.frame = frame

    def predict(self, image):
        starttime = time.time()
        self.predicting = True
        blob = cv2.dnn.blobFromImage(image, self.scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers(self.net))
        class_ids, boxes, confidences = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * 416)
                    center_y = int(detection[1] * 416)
                    w = int(detection[2] * 416)
                    h = int(detection[3] * 416)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        endtime = time.time()
        print('Predictions time {0:.2f} [s] ~ {1:.2f} [fps]'.format(endtime - starttime, 1/(endtime - starttime)))

        while self.drawing:
            pass
        self.class_ids, self.boxes, self.confidences = class_ids, boxes, confidences
        self.predicting = False

    def draw_predictions(self, image):
        self.drawing = True
        indices = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.conf_threshold, self.nms_threshold)
        for i in indices:
            i = i[0]
            box = self.boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_prediction(image, self.class_ids[i], self.confidences[i], round(x), round(y), round(x + w), round(y + h))
        self.drawing = False

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):

        label = str(self.classes[class_id]) + ' {0:.2f}'.format(confidence)
        color = self.COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

    def run(self):
        while True:
            if not self.predicting:
                self.predict(self.frame)


if __name__ == '__main__':
    yolo = Yolov3()
    print('No errors')
