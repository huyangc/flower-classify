#!/usr/bin/env sh

~/software/caffe/build/tools/caffe train \
    --solver=models/GoogleNet/origin/solver.prototxt \
    --weights=/home/zheda/data/models/googlenet.caffemodel -gpu 0
