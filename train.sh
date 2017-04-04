#!/usr/bin/env sh
./caffe/build/install/bin/caffe \
train \
-solver=solver.prototxt \
-weights=../c3d_ucf101_iter_38000.caffemodel \
2>&1 | tee mini_test.log