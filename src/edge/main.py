import shooter as st
import classifier as cl
import detector as dt
import image_utility as util
import cv2 as cv
from PIL import Image
import numpy as np
import pygame
import os

PATH_DETECTOR_MODEL_XML = '../model/face-detection-adas-0001.xml'
PATH_DETECTOR_MODEL_BIN = '../model/face-detection-adas-0001.bin'
PATH_CLASSIFIER_MODEL   = '../model/builtin_mobilenetv2-longrun_edgetpu.tflite'
PATH_CLASSIFIER_LABEL   = '../model/labels.txt'

# raspi screen = 728x480, window size=320x240
# win pos = (728-480)/2, (480-240)/2
os.environ['SDL_VIDEO_WINDOW_POS'] = "204,120"

def main_loop(detector, classifier):
    screen = pygame.display.set_mode((320, 240))
    cam, stream = st.open_camera()
    for frame in cam.capture_continuous(stream, format='bgr', use_video_port=True):
        stream.truncate()
        stream.seek(0)
        process(screen, detector, classifier, stream)
        if(handle_key_event()):
            break

def process(screen, detector, classifier, stream):
    try:
        boxes = process_face_detection(detector, stream.array)
        sentiments = process_sentiment_analysis(classifier, stream.array, boxes)
        draw_results(screen, stream.array, boxes, sentiments)
    except TypeError as err:
        # this error raised sometimes from detector.detect(). and not solved yet
        # so just ignore it an working around
        print(err)
        pass

def process_face_detection(detector, image):
    if image is None or image.shape is None:
        print("invalid argument")
        return None

    # face detection
    detection_results = detector.detect(image)

    # clipped box list, may contain None
    clipped_boxes = []

    for res in detection_results:
        confidence = res[0]
        box        = res[1]

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

def draw_results(screen, image, boxes, sentiments):
    for box, sen in zip(boxes, sentiments):
        cv.rectangle(image, box, color=(0, 255, 0))
        if sen is None:
            continue
        text = "{} {:.1f}%".format(sen[0], sen[1]*100)
        cv.putText(image, text, (box[0], box[1]), cv.FONT_HERSHEY_PLAIN,
            1.2, (255, 255, 255), 1, cv.LINE_AA)

    #cv.imwrite('out.png', image)
    #BGR -> RGB
    rgb_img=image[:,:,::-1]
    pyg_shape = image.shape[1::-1]
    pyg_img = pygame.image.frombuffer(rgb_img.tostring(), pyg_shape, 'RGB')
    screen.blit(pyg_img, (0, 0))
    pygame.display.flip()

def handle_key_event():
    done = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    return done

def main():
    print('app started')
    pygame.init()
    detector = dt.Detector(PATH_DETECTOR_MODEL_XML, PATH_DETECTOR_MODEL_BIN)
    classifier = cl.Classifier(PATH_CLASSIFIER_MODEL, PATH_CLASSIFIER_LABEL)
    main_loop(detector, classifier)
    print('app exit')

if __name__ == '__main__':
  main()
