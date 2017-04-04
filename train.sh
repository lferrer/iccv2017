#!/usr/bin/env sh
./C3D-v1.1/build/install/bin/caffe \
train \
-solver=solver.prototxt \
-weights=../c3d_resnet18_sports1m_r2_iter_2800000.caffemodel \
2>&1 | tee mini_test.log