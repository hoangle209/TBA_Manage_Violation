def iou(box1, box2):
    '''
    format xyxy
    '''
    xmin = max(box1[0], box2[0])
    xmax = min(box1[2], box2[2])
    ymin = max(box1[1], box2[1])
    ymax = min(box1[3], box2[3])

    if xmax <= xmin or ymax <= ymin:
        return 0.

    intersection = (xmax-xmin)*(ymax-ymin)
    s1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    s2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    iou = intersection / (s1+s2-intersection)
    return iou

def custom_nms(bboxes, thresh = 0.8):
    bboxes = sorted(bboxes, key=lambda x:x[-2], reverse=True)
    keep = []
    if not bboxes:
        return 
    keep.append(bboxes[0])
    
    for box1 in bboxes[1:]:
        flag = 0
        for box2 in keep:
            if iou(box1, box2) > thresh:
                flag = 1
                break
        if flag==0:
            keep.append(box1)
    return keep