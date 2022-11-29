from mutils.check_violate_polygon import is_inside_polygon
from mutils.check_violate_rectangle import is_inside_rectangle, find_ref_bb

class CheckViolate:
    '''
      :param vertices: vertices of restricted area: list [list((x1,y1), (x2,y2), ... (xn, yn))]
      :param max_count_violate: maximum number of violations counted to warning
      :param max_numm_track_violate: number of consecutvie frames that still tracking violate
      :param approx_region: approximate bounding rectangle of restricted polygon region
      :param size: if approx_region is True, must provide size of input image
    '''
    def __init__(self, vertices, max_count_violate = 5, max_num_track_violate = 5, approx_region = True, size = None):
        self.vertices_list = vertices
        self.max_count_violate = max_count_violate
        self.max_num_track_violate = max_num_track_violate
        self.approx_region = approx_region
        self.is_inside_polygon = is_inside_polygon
        self.is_inside_rectangle = is_inside_rectangle
        # if self.approx_region:
        self.find_ref_bb = find_ref_bb
        self.size = size
        assert size is not None, print('You have to provide input size of image to use this function')
        self.bounding_rects = [self.find_ref_bb(vertices, size) for vertices in self.vertices_list]
        self.restart()

    def restart(self):
        self.count_violate = [0] * len(self.vertices_list)
        self.keep_track_violate = [0] * len(self.vertices_list)
        self.flag_is_violate = [False] * len(self.vertices_list)


    def is_violate_polygon(self, bboxes, vertices: list, labels: list):
        for bb in bboxes:
          try:
              cx = (bb[0]+bb[2])/2
              cy = (bb[1]+bb[3])/2

              if bb[-1] not in labels and self.is_inside_polygon(vertices, (cx, cy)):
                  print(bb[-1])
                  return True
          except:
              pass
        return False


    def is_violate_rectangle(self, bboxes, vertices, labels):
        for bb in bboxes:
            try:
                cx = (bb[0]+bb[2])/2
                cy = (bb[1]+bb[3])/2

                if bb[-1] not in labels and self.is_inside_rectangle((cx, cy), vertices):
                    return True
            except:
                pass
        return False


    def check(self, is_violate, idx):
        '''
        This function chekcs if a region is violated
          :param is_violate: if at the current frame, the region is violated
          :param idx: id of considered region

          :return whether the region is violated or not
        '''
        if is_violate:
            if self.count_violate[idx] < self.max_count_violate:
                self.count_violate[idx] += 1
            self.flag_is_violate[idx] = True
            self.keep_track_violate[idx] = 0
        else:
            if self.flag_is_violate[idx]:
                if self.keep_track_violate[idx] == self.max_num_track_violate:
                    self.count_violate[idx] = 0
                    self.keep_track_violate[idx] = 0
                    self.flag_is_violate[idx] = False
                else:
                    self.keep_track_violate[idx] += 1

        if self.count_violate[idx] == self.max_count_violate:
            return True
        return False   
    

    def add_vertices(self, vertices):
        if self.approx_region:
            rect = self.find_ref_bb(vertices, self.size)
            self.bounding_rects.append(rect)
        else:
            self.vertices_list.append(vertices)
        self.count_violate.append(0)
        self.keep_track_violate.append(0)
        self.flag_is_violate.append(False)


    def remove_vertices(self, idx):
        if self.approx_region:
            del self.bounding_rects[idx]
        else:
            del self.vertices_list[idx]

        del self.count_violate[idx]
        del self.keep_track_violate[idx]
        del self.flag_is_violate.append[idx]


    def run(self, bboxes, labels: list):
        '''
            :param bboxes: list of list bboxes corresponding to each region needed to manage
            :param labels: list of list labels that are able to work in each region
        '''
        if self.approx_region:
            violate_inside_area = [self.is_violate_rectangle(bbox, rect, label) \
                                                            for (bbox, rect, label) \
                                                            in zip(bboxes, self.bounding_rects, labels)]
        else:
            violate_inside_area = [self.is_violate_polygon(bbox, vertices, label) \
                                                          for (bbox, vertices, label) \
                                                          in zip(bboxes, self.vertices_list, labels)]

        is_violate = list(map(self.check, violate_inside_area, [i for i in range(len(violate_inside_area))]))
        return is_violate