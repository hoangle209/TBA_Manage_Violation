
from inference import _load_model, video_inference_v2
from mutils.read_file import read_file

import argparse
import cv2 as cv

parser = argparse.ArgumentParser(description='TBA Helmets Detection')

parser.add_argument('--weight', type = str, metavar='', help='weight path of the model')
parser.add_argument('--agnostic', action = 'store_true', help='agnostic nms')
parser.add_argument('--conf', type = float, metavar='', help='confidence thresh', default = 0.65)
parser.add_argument('--vertices', type = str, metavar='', help='path to txt storing vertices and classes')

# parser.add_argument('--vertices', action = 'append', metavar='', help='vertices')
# parser.add_argument('--classes', type = list, metavar='', help='classes')

parser.add_argument('--approx', action = 'store_true', help='approximation rectangle around vertices')
parser.add_argument('--robust', action = 'store_true', help='robust detection')
parser.add_argument('--show', action = 'store_true', help='live streaming')
parser.add_argument('--mcv', type = int, metavar='', help='maximum number of violations counted to warning', default=5)
parser.add_argument('--mntv', type = int, metavar='', help='number of maximum consecutive frames that keep tracking violation', default=3)
parser.add_argument('--cam-id', type = str, metavar='', help='camera id')
parser.add_argument('--size-w', type = int, metavar='', help='Input image size w')
parser.add_argument('--size-h', type = int, metavar='', help='Input image size h')
parser.add_argument('--grid-size', type = int, metavar='', help='grid size for tiling detection', default = 576)
parser.add_argument('--save', type = bool, metavar='', help='save video', default=False)

args = parser.parse_args()


def run():
    weight = args.weight
    agnostic = args.agnostic
    conf = args.conf
    model = _load_model(weight, agnostic, conf)

    # TODO - check type that save vertices and classes

    # vertices = args.vertices
    # classes = args.classes

    vertices,classes = read_file(args.vertices)
    
    size = (args.size_h, args.size_w)

    max_count_violate = args.mcv
    max_num_track_violate = args.mntv
    approx = args.approx
    robust = args.robust
    grid_size = args.grid_size
    
    cam_id = args.cam_id
    show = args.show
    save = args.save
    video_inference_v2(cam_id,
                       model,
                       vertices,
                       classes,
                       grid_size,
                       robust,
                       max_count_violate,
                       max_num_track_violate,
                       approx,
                       size,
                       show, 
                       save)


if __name__ == '__main__':
    run()

    

