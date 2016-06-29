DATADIR = '/home/zheda/data/flower_all/102flowers'
ANNOTATIONDIR = '/home/zheda/data/flower_all/102flowers/data/Annotation'
TRAINTXT = 'train_labels_final.txt'
TESTTXT = 'val_labels_final.txt'
CROP_TRAIN_SET = 'crop_train_set'
CROP_TRAIN_SET_256 = 'crop_train_set_256'
MIN_SIZE = 256

from PIL import Image
import os.path as osp
import os

if '__main__':
    crop_all = osp.join(DATADIR,CROP_TRAIN_SET)
    fw = open(osp.join(DATADIR,'crop_train_256.txt'),'w')
    fr = open(osp.join(DATADIR,'cropped_train_all.txt'),'r')
    d = {}
    for line in fr.readlines():
        filename,label = line.split('\t')
        filename = osp.basename(filename)
        d[filename] = label
    for file in os.listdir(crop_all):
        im = Image.open(osp.join(crop_all,file))
        if min(im.size)>=MIN_SIZE:
            fw.write(CROP_TRAIN_SET+'/'+file+'\t'+d[file])