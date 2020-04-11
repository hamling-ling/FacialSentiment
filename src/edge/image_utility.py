import numpy as np

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
    print("pos=", pos, "sz=", sz)
    print("center=", center)
    print("boundary=", boundary)
    max_sz     = np.max(sz)
    outer_sz   = np.array([max_sz, max_sz], dtype=np.int)
    outer_pos =  np.subtract(center,  np.floor_divide(outer_sz, 2))
    print("outer_pos=", outer_pos, "outer_size=", outer_sz)

    negatives = outer_pos < 0
    if (0 < np.sum(negatives)) :
        # negative pos
        print("negative pos")
        return None
    exceeds = (boundary <= outer_pos + outer_sz)
    if (0 < np.sum(exceeds)):
        print("exceeds bound")
        return None
    
    return np.array([outer_pos[0], outer_pos[1], outer_sz[0], outer_sz[1]])
