import torch
import cv2 as cv

from .check_violate import CheckViolate
from Helmets.utils.ious import custom_nms

def _load_model(weight_path, 
                agnostic_nms = False, 
                conf_thresh = 0.65):
                
    model = torch.hub.load('ultralytics/yolov5', 
                           'custom', 
                           path = weight_path)
    model.agnostic = agnostic_nms
    model.conf = conf_thresh
    return model


def _auto_size(ref_rect = None, grid_size = 576):
    rx, ry, rw, rh = ref_rect
    nw = (int(rw/576)+1) * 576
    nh = (int(rh/576)+1) * 576

    offset_x = (nw-rw)/2
    offset_y = (nh-rh)/2

    x = max(int(rx-offset_x), 0)
    y = max(int(ry-offset_y), 0)

    return x, y, nw, nh


def _inference_v2(img, 
                  model, 
                  ref_rect = None, 
                  grid_size = 576, 
                  robust = False):
  
    rx, ry, rw, rh = ref_rect
    _img = img[ry:ry+rh, rx:rx+rw, :]

    _box = []

    # detect in the whole image 
    r = model(_img)
    xyxy = r.xyxyn[0].cpu().numpy()
    for bb in xyxy:
        xmin = int(bb[0] * rw) + rx
        ymin = int(bb[1] * rh) + ry
        xmax = int(bb[2] * rw) + rx
        ymax = int(bb[3] * rh) + ry
        _box.append([xmin, ymin, xmax, ymax, bb[-2], bb[-1]])

    if robust:
        # tile detection
        num_grids_w = int(rw / grid_size)
        num_grids_h = int(rh / grid_size)
        if (num_grids_h > 1 and num_grids_w >= 1) or (num_grids_h >=1 and num_grids_w > 1):
            for k in range(num_grids_w):
                for h in range(num_grids_h):
                    xmin, ymin = k*grid_size, h*grid_size
                    xmax, ymax = (k+1)*grid_size, (h+1)*grid_size

                    img_i = _img[int(ymin):int(ymax), int(xmin):int(xmax), :]
                    r = model(img_i)
                    xyxy = r.xyxyn[0].cpu().numpy()
                    
                    for bb in xyxy:
                        if (bb[2] - bb[0])*(bb[3] - bb[1])*grid_size**2 > 1000: 
                            continue

                        xmin = int((bb[0] + k) * grid_size) + rx
                        ymin = int((bb[1] + h) * grid_size) + ry
                        xmax = int((bb[2] + k) * grid_size) + rx
                        ymax = int((bb[3] + h) * grid_size) + ry
                        _box.append([xmin, ymin, xmax, ymax, bb[-2], bb[-1]])
    _box = custom_nms(_box)
    return _box


# TODO: Multi thread or parallelism

def video_inference_v2(cam_id, 
                       model, 
                       vertices, 
                       classes, 
                       grid_size = 576, 
                       robust = True,
                       max_count_violate=5,
                       max_num_track_violate=5, 
                       approx_region = True, 
                       size = None):
  
    ckv = CheckViolate(vertices, 
                       max_count_violate, 
                       max_num_track_violate, 
                       approx_region, 
                       size)
    
    bounding_rect = ckv.bounding_rect
    bounding_rect = map(_auto_size, bounding_rect)

    bbes = []

    vid = cv.VideoCapture(cam_id)
    while True:
        succ, frame = vid.read()
        if not succ:
            break
        for rect in bounding_rect:
            bb = _inference_v2(frame[...,::-1], model, rect, grid_size, robust)
            bbes += bb
            
        isViolate = ckv.run(bbes, classes)

        # TODO Warning
    
    vid.release()
    cv.destroyAllWindows()