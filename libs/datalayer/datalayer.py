import caffe
import numpy as np
from config.config import cfg
from minibatch import get_minibatch
import yaml

class DataLayer(caffe.Layer):

    def _shuffle_imdb(self):
        self._perm = np.random.permutation(np.arange(len(self._imdb)))
        self._cur = 0


    def _get_next_minibatch_index(self):
        if self._cur+cfg.TRAIN.BATCH_SIZE>=len(self._imdb):
            self._shuffle_imdb()
        ret = self._perm[self._cur:self._cur+cfg.TRAIN.BATCH_SIZE]
        self._cur += cfg.TRAIN.BATCH_SIZE
        return ret

    def _get_next_minibatch(self):
        minibatch_indexes = self._get_next_minibatch_index()
        minibatch_db = [self._imdb[i] for i in minibatch_indexes]
        return get_minibatch(minibatch_db)

    def _prep_imdb(self,imdb):  #imdb = [imagepath,label,boxes,flipped]
        ret = []
        for i in imdb:
            for box in i[2]:
                temp = [i[0], i[1],box,i[3]]
                ret.append(temp)
        return ret



    def set_imdb(self,imdb):
        self._imdb = self._prep_imdb(imdb)
        print 'total parts: ',len(self._imdb)
        self._shuffle_imdb()


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