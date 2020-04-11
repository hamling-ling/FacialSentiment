import cv2 as cv

class Detector():
  def __init__(self, model_xml_path, model_bin_path):
    self.net = cv.dnn_DetectionModel( model_xml_path, model_bin_path)
    self.net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
    
  def detect(self, image):
    _, confidences, boxes = self.net.detect(image, confThreshold=0.5) 
    return confidences, boxes
