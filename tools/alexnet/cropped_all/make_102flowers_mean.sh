#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=~/data/flower_all/102flowers/lmdb
DATA=~/data/flower_all/102flowers
TOOLS=~/software/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/102flowers_cropped_train_lmdb \
  $DATA/102flowers_cropped_train_mean.binaryproto

$TOOLS/compute_image_mean $EXAMPLE/102flowers_val_lmdb \
  $DATA/102flowers_val_mean.binaryproto
echo "Done."
