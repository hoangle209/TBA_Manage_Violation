def tile_inference(img_path, model, size = (1152, 1152), patch_size = 576, conf_thresh = 0.15, save = 'True', save_name = None):
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

    ipath = os.path.split(img_path)[0]
    img = Image.open(img_path).convert('RGB').resize(size)
    _img = np.array(img)

    num_patches_w = int(size[0] / patch_size)
    num_patches_h = int(size[1] / patch_size)
    for k in range(num_patches_w):
        for h in range(num_patches_h):
            xmin, ymin = k*patch_size, h*patch_size
            xmax, ymax = (k+1)*patch_size, (h+1)*patch_size

            img_i = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
            r = model(img_i)
            xyxy = r.xyxyn[0].cpu().numpy()
            
            for bb in xyxy:
                if bb[-2] < conf_thresh: 
                    continue

                xmin = int((bb[0] + k) * patch_size)
                ymin = int((bb[1] + h) * patch_size)
                xmax = int((bb[2] + k) * patch_size)
                ymax = int((bb[3] + h) * patch_size)

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
        _i.save(f'{os.path.join(ipath, name)}.jpg')
        print(f'Save to {ipath} with name as {name}.jpg')
        return True
    return _img     
   
   
def tile_inference_boxes(img_path, model, size = (1152, 1152), patch_size = 576, conf_thresh = 0.15, rtype = 'xyxy'):
    img = Image.open(img_path).convert('RGB').resize(size)
    _img = np.array(img)
    num_patches_w = int(size[0] / patch_size)
    num_patches_h = int(size[1] / patch_size)

    bboxes = []
    for k in range(num_patches_w):
        for h in range(num_patches_h):
            xmin, ymin = k*patch_size, h*patch_size
            xmax, ymax = (k+1)*patch_size, (h+1)*patch_size

            img_i = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
            r = model(img_i)
            xyxy = r.xyxyn[0].cpu().numpy()
            
            for bb in xyxy:
                if bb[-2] < conf_thresh: 
                    continue

                xmin = int((bb[0] + k) * patch_size)
                ymin = int((bb[1] + h) * patch_size)
                xmax = int((bb[2] + k) * patch_size)
                ymax = int((bb[3] + h) * patch_size)
                if rtype == 'xyxy':
                    bboxes.append([xmin, ymin, xmax, ymax, bb[-1]])
                elif rtype == 'xywh':
                    bboxes.append([(xmin + xmax)/2, (ymin + ymax)/2, xmax - xmin, ymax - ymin])
    return bboxes          
