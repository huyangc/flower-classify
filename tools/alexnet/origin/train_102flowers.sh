#!/usr/bin/env sh

~/software/caffe/build/tools/caffe train \
    --solver=/home/zheda/workspace/cnn/flower-classify/models/caffenet/origin/solver.prototxt \
    --weights=/home/zheda/data/models/imagenet_models/ALAX_NET.caffemodel -gpu 0
