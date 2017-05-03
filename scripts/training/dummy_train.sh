#!/usr/bin/env sh
../caffe/build/install/bin/caffe \
train \
-solver=../models/dummy_solver.prototxt \
#2>&1 | tee three_stream.log
