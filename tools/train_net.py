#!/usr/bin/env python

import _init_paths

import caffe
import argparse
import pprint
import numpy as np
import sys
import cv2

from train import train_net,get_training_imdb
from config.config import cfg,get_output_dir
from datasets.imdb import imdb
from datasets.wholeimdb import wholeimage



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a flower-classify network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def _prep_imdb(imdb):  # imdb = [imagepath,label,boxes,flipped]
    ret = []
    for i in imdb:
        for box in i[2]:
            temp = [i[0], i[1], box, i[3]]
            ret.append(temp)
    return ret

def compute_means(imdb):
    imdb = _prep_imdb(imdb)
    num_images = len(imdb)
    processed_ims = []

    for i in xrange(num_images):
        im = cv2.imread(imdb[i][0])  # im.shape = [height,width]
        if imdb[i][3]:
            im = im[:, ::-1, :]
        im = im[imdb[i][2][1]:imdb[i][2][3], imdb[i][2][0]:imdb[i][2][2]]
        print im.shape


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)


    print('Using config:')
    pprint.pprint(cfg)

    # if not args.randomize:
    #     # fix the random seeds (numpy and caffe) for reproducibility
    #     np.random.seed(cfg.RNG_SEED)
    #     caffe.Net.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)
    _imdb = wholeimage('102flowers')
    print 'Loaded datasets `{:s}` for training'.format(_imdb.name)
    _imdb.get_train_image()

    _imdb = get_training_imdb(_imdb)
    imdb = _imdb.train_image

    output_dir = get_output_dir(_imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    # print compute_means(imdb)


    train_net(args.solver, imdb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)


