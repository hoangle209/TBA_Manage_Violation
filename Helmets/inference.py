import torch
import cv2 as cv

def _load_model(weight_path, agnostic_nms = True, conf_thresh = 0.65):
    model = torch.hub.load('ultralytics/yolov5', 
                           'custom', 
                           path = weight_path)
    model.agnostic = agnostic_nms
    model.conf = conf_thresh
    return model


def _inference(img, model, size = (1728, 1152), grid_size = 576, adapt = False):
    if adapt:
        h, w = img.shape[:2]
        nw = int(w / grid_size + 1)*grid_size
        nh = int(h / grid_size + 1)*grid_size
        size = (nw, nh)
      
    img = cv.resize(img, size)
    _box = []

    # detect in the whole image
    r = model(img)
    xyxy = r.xyxyn[0].cpu().numpy()
    for bb in xyxy:
        xmin = int(bb[0] * size[0])
        ymin = int(bb[1] * size[1])
        xmax = int(bb[2] * size[0])
        ymax = int(bb[3] * size[1])
        _box.append([xmin, ymin, xmax, ymax, bb[-2], bb[-1]])

    # tile detection
    num_grids_w = int(size[0] / grid_size)
    num_grids_h = int(size[1] / grid_size)
    for k in range(num_grids_w):
        for h in range(num_grids_h):
            xmin, ymin = k*grid_size, h*grid_size
            xmax, ymax = (k+1)*grid_size, (h+1)*grid_size

            img_i = img[int(ymin):int(ymax), int(xmin):int(xmax), :]
            r = model(img_i)
            xyxy = r.xyxyn[0].cpu().numpy()
            
            for bb in xyxy:
                if (bb[2] - bb[0])*(bb[3] - bb[1])*grid_size**2 > 1500: 
                    continue

                xmin = int((bb[0] + k) * grid_size)
                ymin = int((bb[1] + h) * grid_size)
                xmax = int((bb[2] + k) * grid_size)
                ymax = int((bb[3] + h) * grid_size)
                _box.append([xmin, ymin, xmax, ymax, bb[-2], bb[-1]])
    return img, _box  


def _auto_size(orig_wh, grid_size = 576):
    w, h = orig_wh
    nw = round(w/grid_size) * grid_size
    nh = round(h/grid_size) * grid_size
    return nw, nh

def _inference_v2(img, model, ref_rect, grid_size = 576, robust = True):
    rx, ry, rw, rh = ref_rect
    _img = img[ry:ry+rh, rx:rx+rw, :]
    nw, nh = _auto_size((rw, rh))
    if nw > 0 and nh > 0:
        _img = cv.resize(_img, (nw, nh))

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
        num_grids_w = int(nw / grid_size)
        num_grids_h = int(nh / grid_size)
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

                    xmin = int((bb[0] + k) * grid_size / nw * rw)
                    ymin = int((bb[1] + h) * grid_size / nh * rh)
                    xmax = int((bb[2] + k) * grid_size / nw * rw)
                    ymax = int((bb[3] + h) * grid_size / nh * rh)
                    _box.append([xmin, ymin, xmax, ymax, bb[-2], bb[-1]])
    return _box
