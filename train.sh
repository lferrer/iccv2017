#!/usr/bin/env sh
./caffe/build/install/bin/caffe \
train \
-solver=solver.prototxt --gpu=1 \
-weights=../c3d_ucf101_iter_38000.caffemodel \
2>&1 | tee three_stream_triplet_loss.log
