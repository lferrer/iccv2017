#!/usr/bin/env sh
../../caffe/build/install/bin/caffe \
test \
-model=../../models/two_stream_deploy.prototxt -gpu=7 \
-weights=../../../two_stream_cyclic_loss_iter_60000.caffemodel \
-iterations=1500 \
2>&1 | tee ../../logs/cyclic_loss_features.log

