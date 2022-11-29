from inference import _load_model, video_inference_v2
                      
from mutils.read_file import read_file
from check_violate import CheckViolate
from mutils.ious import custom_nms

import argparse
import time

class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='TBA Helmets Detection')

    def add_args(self):
        self.parser.add_argument('--weight', type = str, metavar='', help='weight path of the model')
        self.parser.add_argument('--agnostic', action = 'store_true', help='agnostic nms')
        self.parser.add_argument('--conf', type = float, metavar='', help='confidence thresh', default = 0.65)
        self.parser.add_argument('--vertices', type = str, metavar='', help='path to txt storing vertices and classes')

        # parser.add_argument('--vertices', action = 'append', metavar='', help='vertices')
        # parser.add_argument('--classes', type = list, metavar='', help='classes')

        self.parser.add_argument('--approx', action = 'store_true', help='approximation rectangle around vertices')
        self.parser.add_argument('--robust', action = 'store_true', help='robust detection')
        self.parser.add_argument('--show', action = 'store_true', help='live streaming')
        self.parser.add_argument('--mcv', type = int, metavar='', help='maximum number of violations counted to warning', default=5)
        self.parser.add_argument('--mntv', type = int, metavar='', help='number of maximum consecutive frames that keep tracking violation', default=3)
        self.parser.add_argument('--cam-id', type = str, metavar='', help='camera id')
        self.parser.add_argument('--size-w', type = int, metavar='', help='Input image size w', default = 1920)
        self.parser.add_argument('--size-h', type = int, metavar='', help='Input image size h', default = 1080)
        self.parser.add_argument('--grid-size', type = int, metavar='', help='grid size for tiling detection', default = 576)
        self.parser.add_argument('--save', action = 'store_true', help='save video')


    def __call__(self):
        self.add_args()
        args = self.parser.parse_args()
        return args


class Interface:
    def __init__(self, args):
        self.load_model = _load_model
        self.video_inference = video_inference_v2
        self.read_file = read_file
        self.args = args

    def run(self):
        weight = self.args.weight
        agnostic = self.args.agnostic
        conf = self.args.conf
        model = self.load_model(weight, agnostic, conf)

        # TODO - check type that save vertices and classes

        # vertices = args.vertices
        # classes = args.classes

        vertices, labels = self.read_file(self.args.vertices)
        
        size = (self.args.size_h, self.args.size_w)

        max_count_violate = self.args.mcv
        max_num_track_violate = self.args.mntv
        approx = self.args.approx
        robust = self.args.robust
        grid_size = self.args.grid_size
        
        cam_id = self.args.cam_id
        show = self.args.show
        save = self.args.save
  
        self.video_inference(cam_id,
                             model,
                             vertices,
                             labels,
                             grid_size,
                             robust,
                             max_count_violate,
                             max_num_track_violate,
                             approx,
                             size,
                             show, 
                             save)

