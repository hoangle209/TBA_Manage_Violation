import torch
import cv2 as cv
import os
import time
import numpy as np

from check_violate import CheckViolate
from mutils.ious import custom_nms

def _load_model(weight_path, 
                agnostic_nms = False, 
                conf_thresh = 0.65):
    '''
    Load Model
    '''    
    model = torch.hub.load('ultralytics/yolov5', 
                           'custom', 
                           path = weight_path)
    model.agnostic = agnostic_nms
    model.conf = conf_thresh
    return model


def _auto_size(ref_rect = None, grid_size = 576):
    '''
    This function approximates dimension size to multiple of grid_size
        :param ref_rect: reference rectangle, [x_top, y_left, w, h]
        :param grid_size: reference size

        :return the bounding rectangle that has size is multiple of grid_size
    '''
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
    '''
    Inference funtion
        :param model: model
        :param ref_rect: do detection only in this region
        :param robust: do tile detection 
        :param grid_size: if robust, divide ref_rect into grid of such size 

        :return bounding box of objects
    '''
    rx, ry, rw, rh = ref_rect
    _img = img[ry:ry+rh, rx:rx+rw, :]

    # check if img's dim is not multiple of grid_size
    if _img.shape[0]%grid_size !=0 or _img.shape[1]%grid_size != 0:
        new_p = np.zeros(shape = (rh, rw), dtype = np.uint8)
        new_p[:_img.shape[0], :_img.shape[1], :] = _img
        _img = new_p

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
    if len(_box) > 1:
        _box = custom_nms(_box)
    return _box


def visual_and_save(_img, _box, label_dict, color_dict, save = 'True', save_name = None):
    if _box:
        for bb in _box:
            xmin, ymin, xmax, ymax = bb[:4]
            cv.rectangle(_img, (xmin, ymin), (xmax, ymax), color_dict[int(bb[-1])], 2)

            textw, texth = cv.getTextSize(f'{label_dict[int(bb[-1])]} {bb[-2]:0.2f}', 
                                          cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv.rectangle(_img, (xmin, max(ymin - texth - 2, 0)), (xmin + textw, ymin), color_dict[int(bb[-1])], -1)

            cv.putText(_img, f'{label_dict[int(bb[-1])]} {bb[-2]:0.2f}', 
                        (xmin, max(ymin-2, 0)), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
    # if save:
    #     _i = Image.fromarray(np.uint8(_img))
    #     name = save_name if save_name is not None else 'img0'
    #     ipath = os.getcwd()
    #     _i.save(f'{os.path.join(ipath, name)}.jpg')
    #     print(f'Save to {ipath} with name as {name}.jpg')
    #     return True
    return _img

# TODO: Multi thread or parallelism

def video_inference_v2(cam_id, 
                       model, 
                       vertices, 
                       labels, 
                       grid_size = 576, 
                       robust = True,
                       max_count_violate=5,
                       max_num_track_violate=5, 
                       approx_region = True, 
                       size = None, 
                       show = False,
                       save = False,
                       path = None):
    '''
    this function checks if restricted areas have violation or not
        :param model: model
        :param vertices: vertices of restricted regions
        :param classes: classes that can enter restricted areas
    '''
    ckv = CheckViolate(vertices, 
                       max_count_violate, 
                       max_num_track_violate, 
                       approx_region, 
                       size)
    
    bounding_rect = ckv.bounding_rects
    bounding_rect = list(map(_auto_size, bounding_rect))

    if save:
        color_dict = {
            0: (0, 138, 0),
            1: (255, 0, 0),
            2: (255, 255, 0),
            3: (255, 255, 255),
            4: (0, 0, 255),
            5: (255, 165, 0), 
            6: (255, 0, 255)
        }
        label_dict = {
            0: 'None',
            1: 'Red',
            2: 'Yellow',
            3: 'White',
            4: 'Blue',
            5: 'Orange', 
            6: 'Others'
        }

        name = cam_id.split(os.sep)[-1]
        name = name.split('.')[0]
        spath = os.path.join(path, name) if path is not None \
                    else os.path.join(os.getcwd(), f'{name}.avi')
        print(spath)
        writer = cv.VideoWriter(spath,
                                cv.VideoWriter_fourcc(*'MJPG'),
                                15, (size[1], size[0]))

    vid = cv.VideoCapture(cam_id)
    while True:
        succ, frame = vid.read()
        if not succ:
            break

        # FPS
        begin = time.time()

        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        bbes = []
        for rect in bounding_rect:
            bb = _inference_v2(img, 
                               model, 
                               rect, 
                               grid_size, 
                               robust) 
            bbes.append(bb)

        isViolate = ckv.run(bbes, labels)

        t = time.time() - begin
        print(f'Print something: {isViolate} - FPS: {1/t}')
        
        # if show:
            # TODO show

        # TODO Warning

        # TODO save
        if save:
            for v, r, bb in zip(isViolate, ckv.bounding_rects, bbes):
                visual_and_save(img, bb, label_dict, color_dict, False)
                x,y,w,h = r

                c = (255, 0, 0) if v else (0, 255, 0)

                cv.rectangle(img, (x,y), (x+w, y+h), c, 3)
                cv.putText(img, f'{v}', 
                           (x, y), 
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            writer.write(img[...,::-1])
    
    if save:
        writer.release()
    vid.release()
    cv.destroyAllWindows()