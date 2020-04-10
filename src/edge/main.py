import shooter as st
import classifier as cl
import detector as dt

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
    # face detection
    confidences, boxes = detector.detect(stream.array)
    for confidence, box in zip(list(confidences), boxes):
        print(confidence, box)

    # sengiment analysis
    # t.b.d

def main():
    print('app started')
    detector = dt.Detector(PATH_DETECTOR_MODEL_XML, PATH_DETECTOR_MODEL_BIN)
    classifier = cl.Classifier(PATH_CLASSIFIER_MODEL, PATH_CLASSIFIER_LABEL)
    main_loop(detector, classifier)
    print('app exit')

if __name__ == '__main__':
  main()
