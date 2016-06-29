#!/usr/bin/env sh

~/software/caffe/build/tools/caffe train \
    --solver=/home/zheda/workspace/cnn/flower-classify/models/VGG16/origin/solver_vgg.prototxt \
    --weights=/home/zheda/data/models/imagenet_models/VGG16.v2.caffemodel -gpu 1
