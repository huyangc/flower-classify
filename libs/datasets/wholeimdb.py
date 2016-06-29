from datasets.imdb import imdb
import os.path as osp
import cPickle

PICKLE_SUFFIX = ".pkl"
DATADIR = '/home/zheda/data/flower_all/102flowers'
ANNOTATIONDIR = '/home/zheda/data/flower_all/102flowers/data/Annotation'
TRAINTXT = 'train_labels_final.txt'
TESTTXT = 'val_labels_final.txt'

class wholeimage(object):

    def __init__(self, name):
        self.train_image = []  # [imagename,label,flipped]
        self.test_image = []
        self._name = name

    @property
    def name(self):
        return self._name

    def get_train_image(self, datadir=DATADIR, traintxt=TRAINTXT):
        self._prep_image(datadir,traintxt, True)
        return self.train_image

    def get_test_image(self, datadir=DATADIR, testtxt=TRAINTXT):
        self._prep_image(datadir,testtxt, False)
        return self.test_image

    def _prep_image(self, datadir, imagefile,train):  # datadir: path to training data folder, traintxt: training dataset including filenames and labels
        opera = []
        with open(osp.join(datadir, imagefile), 'r') as fr:
            for line in fr.readlines():
                filename, label = line.split('\t')  # filename: filename.jpg\t1
                templist = []
                templist.append(osp.join(datadir, 'data', filename))
                templist.append(int(label)-1)
                templist.append(False)
                opera.append(templist)
        if train:
            self.train_image = opera
        else:
            self.test_image = opera

    def append_flipped_images(self, training_set=True):
        if training_set:
            self._append_flipped_images(self.train_image)
        else:
            self._append_flipped_images(self.test_image)

    def _append_flipped_images(self, imageset):
        num_images = len(imageset)

        for i in xrange(num_images):
            imageset.append([imageset[i][0], imageset[i][1], True])