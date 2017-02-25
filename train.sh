#!/usr/bin/env sh
set -e

/home/lferrer/Downloads/caffe/build/tools/caffe \
  train \
  --solver=solver.prototxt \
  $@ \
  2>&1 | tee min_test.log
