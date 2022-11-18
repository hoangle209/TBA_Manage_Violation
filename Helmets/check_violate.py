from Helmets.utils.check_violate_polygon import is_inside_polygon
from Helmets.utils.check_violate_rectangle import is_inside_rectangle, find_ref_bb

class CheckViolate:
    '''
      :param vertices: vertices of restricted area
      :param max_count_violate: maximum number of violations counted to warning
      :param max_numm_track_violate: number of consecutvie frames that still tracking violate
      :param approx_region: approximate bounding rectangle of restricted polygon region
      :param size: if approx_region is True, must provide size of input image
    '''
    def __init__(self, vertices, max_count_violate = 5, max_num_track_violate = 5, approx_region = True, size = None):
        self.restart()
        self.vertices = vertices
        self.max_count_violate = max_count_violate
        self.max_num_track_violate = max_num_track_violate
        self.approx_region = approx_region
        self.is_inside_polygon = is_inside_polygon
        self.is_inside_rectangle = is_inside_rectangle
        if self.approx_region:
            assert size is not None, print('You have to provide input size of image to use this function')
            self.bb_rect = find_ref_bb(vertices, size)

    def restart(self):
        self.count_violate = 0
        self.keep_track_violate = 0
        self.flag_is_violate = False


    def is_violate_polygon(self, bboxes, vertices: list, labels: list):
        for bb in bboxes:
            cx = (bb[0]+bb[2])/2
            cy = (bb[1]+bb[3])/2

            # if bb[-1] in labels and not is_inside_polygon(restricted_area, (cx, cy)):
            #     return 1
            if bb[-1] not in labels and self.is_inside_polygon(vertices, (cx, cy)):
                return True
            else:
                continue
        return False


    def is_violate_rectangle(self, bboxes, vertices, labels):
        for bb in bboxes:
            cx = (bb[0]+bb[2])/2
            cy = (bb[1]+bb[3])/2

            if bb[-1] not in labels and self.is_inside_rectangle((cx, cy), vertices):
                return True
        return False


    def check(self, is_violate):
        if is_violate:
            if self.count_violate < self.max_count_violate:
                self.count_violate += 1
            self.flag_is_violate = True
        else:
            if self.flag_is_violate:
                if self.keep_track_violate == self.max_num_track_violate:
                    self.restart()
                else:
                    self.keep_track_violate += 1

        if self.count_violate == self.max_count_violate:
            return True
        return False   
    

    def run(self, bboxes, labels: list):
        if self.approx_region:
            violate_inside_area = self.is_violate_rectangle(bboxes, self.bb_rect, labels)
        else:
            violate_inside_area = self.is_violate_polygon(bboxes, self.vertices, labels)

        is_violate = self.check(violate_inside_area)
        return is_violate