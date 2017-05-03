#!/usr/bin/env sh
../../caffe/build/install/bin/caffe \
train \
-solver=../../models/triplet_loss_solver.prototxt --gpu=2 \
-weights=../../../c3d_ucf101_iter_38000.caffemodel \
2>&1 | tee three_stream_triplet_loss.log
