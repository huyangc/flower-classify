import cv2
from datalayer.minibatch import im_list_to_blob
from config.config import cfg
import numpy as np

def _get_blob(im,box):
    blobs = {'data':None}
    processed_imgs = []
    im = im.astype(np.float32,copy=False)
    im -= cfg.PIXEL_MEANS
    im = im[box[1]:box[3],box[0]:box[2]]
    im = cv2.resize(im,(cfg.TEST.TARGET_SIZE[0],cfg.TEST.TARGET_SIZE[1]))
    processed_imgs.append(im)
    blobs['data'] = im_list_to_blob(processed_imgs)
    return blobs

import matplotlib.pyplot as plt
from PIL import Image
def showboxes(imdb):
    boxes = imdb[2]
    img = Image.open(imdb[0])
    size = len(boxes)
    cols = 10
    rows = size / cols + 1

    f, axarr = plt.subplots(rows, cols)
    i, j = 0, 0
    for box in boxes:
        i += j / cols
        j = j % cols
        image = img.crop(box)

        # segments_fzt = segments_fz[box[0][1]:box[0][3],box[0][0]:box[0][2]]
        # plt.imshow(mark_boundaries(image, segments_fzt))
        axarr[i, j].imshow(image)
        axarr[i, j].axis('off')
        j += 1
    plt.show()

def im_classify(net, imdb,truepic,all):
    im = cv2.imread(imdb[0])
    ret_temp = {}
    boxes = []
    if not all:
        for box in imdb[2]:
            if min((box[2] - box[0]), (box[3] - box[1])) >= 256:
                boxes.append(box)
    else:
        boxes = imdb[2]
    if len(boxes) == 0:
        boxes.append([0,0,im.shape[1],im.shape[0]])
    for box in boxes:

        blob = _get_blob(im,box)
        net.blobs['data'].reshape(*(blob['data'].shape))
        blobs_out = net.forward(data=blob['data'].astype(np.float32,copy=False))
        key = blobs_out['prob'].argmax()
        ret_temp[key] = ret_temp.get(key,0) + blobs_out['prob'][0][key]
    ret_temp = sorted(ret_temp.iteritems(),key=lambda d:d[1],reverse=True)
    if ret_temp[0][0] == imdb[1]:
        return 1
    else:
        # showboxes(imdb)
        print imdb[0]
        print imdb[1], ret_temp, boxes
        return 0



def test_net(net, imdb, all = True):
    num_images = len(imdb)

    test_labels = [0 for _ in xrange(num_images)]
    labels = [imdb[i][1] for i in xrange(num_images)]
    truepic = 0
    for i in xrange(num_images):
        test_labels[i] = im_classify(net, imdb[i],truepic,all)
    assert len(test_labels) == len(labels)
    print float(sum(test_labels))/float(num_images)