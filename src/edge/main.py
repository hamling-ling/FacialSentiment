import shooter as st
import classifier as cl
import detector as dt
import image_utility as util
import cv2 as cv
from PIL import Image
import numpy as np

PATH_DETECTOR_MODEL_XML = 'models/face-detection-adas-0001.xml'
PATH_DETECTOR_MODEL_BIN = 'models/face-detection-adas-0001.bin'
PATH_CLASSIFIER_MODEL   = 'models/builtin_mobilenetv2-longrun_edgetpu.tflite'
PATH_CLASSIFIER_LABEL   = 'models/labels.txt'


def main_loop(detector, classifier):
    cam, stream = st.open_camera()
    for frame in cam.capture_continuous(stream, format='bgr', use_video_port=True):
        stream.truncate()
        stream.seek(0)
        process(detector, classifier, stream)
        break

def process(detector, classifier, stream):
    boxes = process_face_detection(detector, stream.array)

    sentiments = process_sentiment_analysis(classifier, stream.array, boxes)

    draw_results(stream.array, boxes, sentiments)

def process_face_detection(detector, image):
    # face detection
    detection_results = detector.detect(image)

    # clipped box list, may contain None
    clipped_boxes = []

    for res in detection_results:
        confidence = res[0]
        box        = res[1]
        print(confidence, box)

        clipped = util.clip_box(box, image.shape)
        if clipped is not None:
            cv.rectangle(image, clipped, color=(0, 0, 255))
            clipped_boxes.append(clipped)

    return clipped_boxes

def process_sentiment_analysis(classifier, image, boxes):
    if(len(boxes) == 0):
        return None, None

    ret = []
    for box in boxes:
        img = util.cropToGray(image, box, 32, classifier.get_input_size())

        # debug
        #img.save("input_face.png", format='PNG')

        result = classifier.predict(img)
        ret.append(result)
    return ret

def draw_results(image, boxes, sentiments):
    print("draw_result ", len(boxes), len(sentiments))
    for box, sen in zip(boxes, sentiments):
        cv.rectangle(image, box, color=(0, 255, 0))
        if sen is None:
            continue
        text = "{} {:.1f}%".format(sen[0], sen[1]*100)
        print(text)
        cv.putText(image, text, (box[0], box[1]), cv.FONT_HERSHEY_PLAIN,
            0.8, (255, 255, 255), 1, cv.LINE_AA)

    cv.imwrite('out.png', image)

def main():
    print('app started')
    detector = dt.Detector(PATH_DETECTOR_MODEL_XML, PATH_DETECTOR_MODEL_BIN)
    classifier = cl.Classifier(PATH_CLASSIFIER_MODEL, PATH_CLASSIFIER_LABEL)
    main_loop(detector, classifier)
    print('app exit')

if __name__ == '__main__':
  main()
