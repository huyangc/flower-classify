import cv2
import cPickle
DATADIR = '/home/zheda/data/flower_all/102flowers'
ANNOTATIONDIR = '/home/zheda/data/flower_all/102flowers/data/Annotation'
TRAINTXT = 'train_labels_final.txt'
TESTTXT = 'val_labels_final.txt'

def _prep_imdb(imdb):  # imdb = [imagepath,label,boxes,flipped]
    ret = []
    for i in imdb:
        for box in i[2]:
            temp = [i[0], i[1], box, i[3]]
            ret.append(temp)
    return ret

temp = cPickle.load(open('../_home_zheda_data_flower_all_102flowers_train_labels_final.txt.pkl','rb'))
print temp[0]
print len(temp)

