import torch
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os

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


def _inference(img, model, size = (1152, 1152), grid_size = 576, conf_thresh = 0.15, adapt = False):
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
        if bb[-2] < conf_thresh: 
            continue
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
                if bb[-2] < conf_thresh or (bb[2] - bb[0])*(bb[3] - bb[1])*grid_size**2 > 1000: 
                    continue

                xmin = int((bb[0] + k) * grid_size)
                ymin = int((bb[1] + h) * grid_size)
                xmax = int((bb[2] + k) * grid_size)
                ymax = int((bb[3] + h) * grid_size)
                _box.append([xmin, ymin, xmax, ymax, bb[-2], bb[-1]])

    _box = custom_nms(_box)
    return img, _box  
 

def visual_and_save(_img, _box, save = 'True', save_name = None):
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
                
    if save:
        _i = Image.fromarray(np.uint8(_img))
        name = save_name if save_name is not None else 'img0'
        ipath = os.getcwd()
        _i.save(f'{os.path.join(ipath, name)}.jpg')
        print(f'Save to {ipath} with name as {name}.jpg')
        return True
    return _img
  
  
def video_inference(cam_id, model, size = (1152, 1152), grid_size = 576, conf_thresh = 0.15, adapt = False):
    vid = cv.VideoCapture(cam_id)
    count = 1
    while True:
        ret, cap = vid.read()
        _img = cap.copy()[..., ::-1]
        copy, bb = _inference(_img, model, size, grid_size, conf_thresh, adapt)
        i = visual_and_save(copy, bb, save=False)
        cv2_imshow(i[..., ::-1])
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        count+=1
    cv.release()
    cv.destroyAllWindows()
