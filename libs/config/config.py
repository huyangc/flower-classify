import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE =64

__C.TRAIN.USE_FLIPPED = True

__C.TRAIN.USE_PREFETCH=False

__C.TRAIN.TARGET_SIZE=[227,227]

__C.TRAIN.SNAPSHOT_ITERS = 10000

__C.TRAIN.MAX_ITERATION=100000

__C.TRAIN.SNAPSHOT_INFIX = ''

__C.TEST = edict()

__C.TEST.TARGET_SIZE = [227, 227]
__C.RNG_SEED = 3
__C.MIN_BOX=256
__C.EPS = 1e-14

__C.PIXEL_MEANS=[[[102.9801,115.9465,122.7717]]]

__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__),'..','..'))

__C.EXP_DIR = 'default'


def get_output_dir(imdb,net):
    path = osp.abspath(osp.join(__C.ROOT_DIR,'output',__C.EXP_DIR,imdb.name))
    return path


