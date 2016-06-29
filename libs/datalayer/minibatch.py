import cv2
import numpy as np
from config.config import cfg
def get_minibatch(imdb):    #imdb: [image files path,label,boxes,flipped]
    num_images = len(imdb)
    processed_ims = []

    for i in xrange(num_images):
        im = cv2.imread(imdb[i][0])    #im.shape = [height,width]
        if imdb[i][3]:
            im = im[:,::-1,:]
        im = im[imdb[i][2][1]:imdb[i][2][3],imdb[i][2][0]:imdb[i][2][2]]
        im = prep_im_for_blob(im,cfg.PIXEL_MEANS,cfg.TRAIN.TARGET_SIZE)
        processed_ims.append(im)
    im_blob = im_list_to_blob(processed_ims)
    labels_blob = np.zeros((0),dtype=np.float32)
    labels = [imdb[i][1] for i in xrange(num_images)]
    labels_blob = np.hstack((labels_blob,labels))
    blobs = {"data":im_blob,
             'labels':labels_blob}
    return blobs



def im_list_to_blob(ims):
    assert len(ims) != 0
    shape = ims[0].shape
    num_images = len(ims)

    blob = np.zeros((num_images,shape[0],shape[1],3),dtype=np.float32)
    for i in xrange(num_images):
        blob[i] = ims[i]

    channel_swap = (0,3,1,2) #caffe blob n,k,h,w
    blob = blob.transpose(channel_swap)
    return blob



def prep_im_for_blob(im,pixel_means,target_size):  #target size is a [height, width]
    im = im.astype(np.float32,copy=False)
    im -= pixel_means
    # im_shape = im.shape
    im = cv2.resize(im,(target_size[0],target_size[1]))

    return im

