#!/usr/bin/env sh

~/software/caffe/build/tools/caffe train \
    --solver=models/GoogleNet/solver.prototxt \
    --snapshot=output/googlenet/102flowers_iter_81922.solverstate -gpu 1
