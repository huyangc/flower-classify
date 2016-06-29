#!/usr/bin/env sh

~/software/caffe/build/tools/caffe train \
    --solver=solver.prototxt \
    --snapshot=/home/zheda/workspace/cnn/flower-classify/102flowers_iter_81922.solverstate -gpu 1
