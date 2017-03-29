#!/usr/bin/env sh
./c3d/C3D-v1.1/build/install/bin/caffe \
train \
--solver=solver.prototxt \
--weights=../conv3d_deepnetA_sport1m_iter_1900000 \
2>&1 | tee mini_test.log