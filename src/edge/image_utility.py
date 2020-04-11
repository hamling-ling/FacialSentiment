import numpy as np
import cv2 as cv

def cvbound2bound(cvbound):
    boundary = np.zeros(2, dtype=np.int)
    boundary[0] = cvbound[1]
    boundary[1] = cvbound[0]
    return boundary

def clip_box(box, cvbound):
    boundary = cvbound2bound(cvbound)
    
    pos = box[0:2]
    sz   = box[2:4]
    center    = np.add(pos, np.floor_divide(sz,2))

    max_sz     = np.max(sz)
    outer_sz   = np.array([max_sz, max_sz], dtype=np.int)
    outer_pos =  np.subtract(center,  np.floor_divide(outer_sz, 2))

    negatives = outer_pos < 0
    if (0 < np.sum(negatives)) :
        # negative pos
        return None
    
    exceeds = (boundary <= outer_pos + outer_sz)
    if (0 < np.sum(exceeds)):
        return None
    
    return np.array([outer_pos[0], outer_pos[1], outer_sz[0], outer_sz[1]])

def cropToGray(image, box, min_size, dest_size):
    x, y, w, h = box[0], box[1], box[2], box[3]
    if( w < min_size or y < min_size):
        return None

    clipped_image = image[y:y+h, x:x+w, :]
    img_gray = cv.cvtColor(clipped_image, cv.COLOR_BGR2GRAY)

    # resized PIL Image
    pil_gray = cv.resize(img_gray , dest_size)
    return pil_gray