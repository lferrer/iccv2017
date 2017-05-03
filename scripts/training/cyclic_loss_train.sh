#!/usr/bin/env sh
../../caffe/build/install/bin/caffe \
train \
-solver=../../models/cyclic_loss_solver.prototxt --gpu=0 \
-weights=../../../c3d_ucf101_iter_38000.caffemodel \
2>&1 | tee ../../logs/three_stream_cyclic_loss.log
