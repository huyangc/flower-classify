import cv2
import numpy as np
from config.config import cfg
import matplotlib.pyplot as plt
def get_minibatch(imdb,iteration=0):    #imdb: [image,label]
    num_images = len(imdb)
    # print 'num_images: batch_size:',num_images
    processed_ims = []

    for i in xrange(num_images):
        im = imdb[i][0]
        im = prep_im_for_blob(im,cfg.PIXEL_MEANS,cfg.TRAIN.TARGET_SIZE)
        processed_ims.append(im)

    # for im in processed_ims:
    #     plt.imshow(im)
    #     plt.show()
    # print 'processimg_size:',len(processed_ims)
    im_blob = im_list_to_blob(processed_ims)
    labels_blob = np.zeros((0),dtype=np.float32)
    labels = [int(imdb[i][1]) for i in xrange(num_images)]
    labels_blob = np.hstack((labels_blob,labels))
    assert len(labels_blob) == len(im_blob)
    blobs = {"data":im_blob,
             'labels':labels_blob}
    # for im in blobs['data']:
    #     channel_swap = (1,2,0)
    #     im = im.transpose(channel_swap)
    #     print im.shape
    #
    #     plt.imshow(im)
    #     plt.show()
    return blobs



def im_list_to_blob(ims):
    assert len(ims) != 0
    shape = ims[0].shape
    num_images = len(ims)

    blob = np.zeros((num_images,shape[0],shape[1],3),dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    channel_swap = (0,3,1,2) #caffe blob n,k,h,w
    blob = blob.transpose(channel_swap)
    return blob



def prep_im_for_blob(im,pixel_means,target_size):  #target size is a [height, width]
    im = im.astype(np.float32,copy=False)
    im -= pixel_means
    # im_shape = im.shape
    im = cv2.resize(im,(target_size[0],target_size[1]))

    return im

