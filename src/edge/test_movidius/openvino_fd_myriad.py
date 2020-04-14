import cv2 as cv

net = cv.dnn_DetectionModel('../../model/face-detection-adas-0001.xml',
                            '../../model/face-detection-adas-0001.bin')
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

frame = cv.imread('../../sample/sample.bmp')
if frame is None:
    raise Exception('Image not found')

_, confidences, boxes = net.detect(frame, confThreshold=0.5)

for confidence, box in zip(list(confidences), boxes):
    cv.rectangle(frame, box, color=(0, 255, 0))
    print(confidence, box)

cv.imwrite('out.png', frame)
