import numpy as np
import cv2 as cv

def find_ref_bb(vertices, size):
    bg = np.zeros(shape = (size[0], size[1]), dtype = np.uint8)
    bg = cv.polylines(bg, [np.array(vertices, dtype = np.int32)],
                      True, (255), 1)
    # cv.fillPoly(bg, [np.array(vertices, dtype = np.int32)], (255))

    c, _ = cv.findContours(bg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cnt = next(iter(c))
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02*peri, True)
    xl,yl,w,h = cv.boundingRect(approx)
    return xl, yl, w, h

def is_inside_rectangle(point, vertices):
    xl, yl, w, h = vertices
    x, y = point
    if x < xl+w/2 and x > xl-w/2 and y < y+h/2 and y > y-h/2:
        return True
    return False