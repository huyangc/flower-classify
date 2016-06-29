# image database
import os
import os.path as osp
import cPickle
from xml.dom.minidom import Document
import xml.dom.minidom as xdm
import PIL
import copy
from PIL import Image
PICKLE_SUFFIX = ".pkl"

DATADIR = '/home/zheda/data/flower_all/102flowers'
ANNOTATIONDIR = '/home/zheda/data/flower_all/102flowers/data/Annotation'
TRAINTXT = 'train_labels_final.txt'
TESTTXT = 'val_labels_final.txt'
CROP_TRAIN_SET = 'crop_train_set'

class imdb(object):

    def __init__(self,name):
        self.train_image = []  #[imagename,label,boxes,flipped]
        self.test_image = []
        self._name = name


    def get_train_image(self,datadir=DATADIR,annotationdir=ANNOTATIONDIR,traintxt=TRAINTXT):
        self._prep_image(datadir,annotationdir,traintxt,True)
        return self.train_image

    def get_test_image(self,datadir=DATADIR,annotationdir=ANNOTATIONDIR,testtxt=TRAINTXT):
        self._prep_image(datadir,annotationdir,testtxt,False)
        return self.test_image

    @property
    def name(self):
        return self._name

    def load_boxes(self,xmlfilepath):
        dom = xdm.parse(xmlfilepath)
        rootDoc = dom.documentElement
        ret = []
        # for bndboxNode in rootDoc.getElementsByTagName('bndbox'):
        #     temp = []
        #     for i in range(1,8,2):
        #         temp.append(int(bndboxNode.childNodes[i].childNodes[0].nodeValue))
        #     ret.append(temp)
        for objectNode in rootDoc.getElementsByTagName('object'):
            name = objectNode.getElementsByTagName('name')[0].childNodes[0].nodeValue
            if name == 'flower':
                bndboxNode = objectNode.getElementsByTagName('bndbox')[0]
                temp = []
                for i in range(1, 8, 2):
                    temp.append(int(bndboxNode.childNodes[i].childNodes[0].nodeValue))
                ret.append(temp)
        return ret

    def append_flipped_images(self,training_set=True):

        if training_set:
            self._append_flipped_images(self.train_image)
        else:
            self._append_flipped_images(self.test_image)

    def _append_flipped_images(self,imageset):
        num_images = len(imageset)
        widths = [PIL.Image.open(imageset[i][0]).size[0] for i in xrange(num_images)]
        for i in xrange(num_images):
            width = widths[i]
            boxnums = len(imageset[i][2])
            tempbox = []
            for j in xrange(boxnums):
                box = imageset[i][2][j]
                boxa = copy.copy(box)
                oldx1 = box[0]
                oldx2 = box[2]
                boxa[0] = width-oldx2
                boxa[2] = width-oldx1
                assert box[2]>box[0]
                tempbox.append(boxa)
            assert len(tempbox) == boxnums
            imageset.append([imageset[i][0],imageset[i][1],tempbox,True])
            # boxes = imageset[:,2].copy()
            # oldx1 = boxes[:, 0].copy()
            # oldx2 = boxes[:, 2].copy()
            # boxes[:, 0] = widths[i] - oldx2 - 1
            # boxes[:, 2] = widths[i] - oldx1 - 1
            # assert (boxes[:, 2] >= boxes[:, 0]).all()
            # imageset.append([imageset[i][0], imageset[i][1], boxes,True])


    '''
      traintxt is under datadir, so in this code, I use osp.join(datadir,traintxt) to generate the final traintxt path
      annotationdir is also the same as datadir, it is the prefix of the annotation xml file path

      datadir include the train.txt, test.txt and image files all together.
      -datadir
        -train.txt.
        -test.txt
        -data
            xxxx.jpg
    '''
    def _prep_image(self, datadir, annotationdir, imagefile, train):    #datadir: path to training data folder, traintxt: training dataset including filenames and labels
        opera = []

        picklefile = osp.join(datadir, imagefile)
        picklefile = picklefile.replace('/','_')+PICKLE_SUFFIX
        if osp.exists(picklefile):
            if train:
                self.train_image = cPickle.load(open(picklefile,'rb'))
            else:
                self.test_image = cPickle.load(open(picklefile,'rb'))
            return
        with open(osp.join(datadir, imagefile), 'r') as fr:
            for line in fr.readlines():
                filename, label = line.split('\t')    #filename: filename.jpg\t1
                templist = []

                xmlfile = filename.split(".")[0]+".xml"
                xmlfilepath = osp.join(annotationdir,xmlfile)
                boxes = self.load_boxes(xmlfilepath)

                templist.append(osp.join(datadir,'data',filename))
                templist.append(int(label))
                templist.append(boxes)
                templist.append(False)
                opera.append(templist)
        cPickle.dump(opera,open(picklefile,'wb'))
        if train:
            self.train_image = opera
        else:
            self.test_image = opera

if '__main__':
    imdb = imdb('102flower')
    imgs = imdb.get_train_image()
    cropped_train_path = osp.join(DATADIR,CROP_TRAIN_SET)
    if not osp.exists(cropped_train_path):
        os.mkdir(cropped_train_path)
    fw = open(osp.join(DATADIR,'cropped_train_all.txt'),'w')
    for img in imgs:
        imgfile = img[0]
        image = Image.open(imgfile)
        base_name,ext = osp.splitext(imgfile)
        base_name = osp.basename(base_name)
        for i,box in enumerate(img[2]):
            im = image.crop(box)
            filename = base_name+'_'+str(i)+ext
            print filename
            # im.save(osp.join(DATADIR,CROP_TRAIN_SET,filename))
            fw.write(CROP_TRAIN_SET+'/'+filename+'\t'+str(img[1]-1)+'\n')
    fw.close()