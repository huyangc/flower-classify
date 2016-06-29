import caffe
import numpy as np
from config.config import cfg
from wminibatch import get_minibatch
import os.path as osp
import cPickle

PICKLE_SUFFIX = '.pkl'

PICKLE_FILE = '102flower_whole.pkl'
import cv2
import yaml

class DataLayer(caffe.Layer):

    def _shuffle_imdb(self):
        self._perm = np.random.permutation(np.arange(len(self._imdb)))
        self._cur = 0
        self._iteration = 0

    def _get_next_minibatch_index(self):
        if self._cur+cfg.TRAIN.BATCH_SIZE>len(self._imdb):
            self._shuffle_imdb()
        ret = self._perm[self._cur:self._cur+cfg.TRAIN.BATCH_SIZE]
        self._cur += cfg.TRAIN.BATCH_SIZE
        return ret

    def _get_next_minibatch(self):
        minibatch_indexes = self._get_next_minibatch_index()
        minibatch_db = [self._imdb[i] for i in minibatch_indexes]
        self._iteration+=1
        return get_minibatch(minibatch_db,self._iteration)





    def set_imdb(self,imdb):
        self._imdb = self._prep_imdb(imdb)
        self._shuffle_imdb()

    def _prep_imdb(self,imdb):
        # if osp.exists(PICKLE_FILE):
        #     return cPickle.load(open(PICKLE_FILE,'rb'))

        num_images = len(imdb)
        # print 'num_images: batch_size:',num_images
        processed_ims = []

        for i in xrange(num_images):
            im = cv2.imread(imdb[i][0])  # im.shape = [height,width]
            if imdb[i][2]:
                im = im[:, ::-1, :]
            processed_ims.append([im,imdb[i][1]])

        # cPickle.dump(processed_ims,open(PICKLE_FILE,'wb'))
        return processed_ims

    def setup(self, bottom, top):
        self._name_to_top_map = {
            'data':0,
            'labels':1
        }
        top[0].reshape(cfg.TRAIN.BATCH_SIZE,3,cfg.TRAIN.TARGET_SIZE[0],cfg.TRAIN.TARGET_SIZE[1])
        top[1].reshape(cfg.TRAIN.BATCH_SIZE)


    def forward(self, bottom, top):
        blobs = self._get_next_minibatch()
        for blob_name,blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            top[top_ind].reshape(*(blob.shape))
            top[top_ind].data[...] = blob.astype(np.float32,copy=False)

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

class BatchLoader(object):
    hello = 1