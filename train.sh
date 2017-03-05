#!/usr/bin/env sh
set -e

~/c3d/C3D-v1.1/build/install/bin/caffe \
  train \
  --solver=solver.prototxt --gpu=1
  --weights=../conv3d_deepnetA_sport1m_iter_1900000?dl=0 \
  $@ \
  2>&1 | tee min_test.log
