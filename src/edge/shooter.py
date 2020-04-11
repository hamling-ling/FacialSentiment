import picamera
from picamera.array import PiRGBArray
import time

CAMERA_WIDTH  = 320
CAMERA_HEIGHT = 240

def open_camera():
    camera            = picamera.PiCamera()
    camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)
    camera.led        = True
    camera.framerate  = 32
    camera.hflip      = False
    camera.vflip      = True
    rawCap            = PiRGBArray(camera, size=(CAMERA_WIDTH, CAMERA_HEIGHT))

    #warm up
    time.sleep(0.1)

    return camera, rawCap

def close_camera(camera):
    camera.close()
    camera.led = False
