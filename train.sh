#!/usr/bin/env sh
set -e

/home/lferrer/Downloads/caffe/build/tools/caffe \
  train \
  --solver=c3d/solver.prototxt \
  $@ \
  2>&1 | tee c3d/c3d_test.log
