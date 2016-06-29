
"""Test a flower classify network on an image database."""

import _init_paths
from config.config import cfg
from datasets.imdb import imdb
from test import test_net
import caffe
import argparse
import pprint
import time, os, sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--part',dest='part',
                        help='all part or not',
                        default=True,type=bool)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    print('Using config:')
    pprint.pprint(cfg)



    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = imdb('102flowers_part_256')
    _imdb = imdb.get_test_image(wholepic=False)
    print 'nums of images:',len(_imdb)
    test_net(net,_imdb,all=False)    #_imdb[imagepath,label,boxes,flipped]
