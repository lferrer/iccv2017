#!/usr/bin/env sh
../../caffe/build/install/bin/caffe \
test \
-model=../../models/three_stream_deploy.prototxt -gpu=7 \
-weights=../../weights/three_stream_triplet_loss_iter_60000.caffemodel \
-iterations=1000 \
2>&1 | tee ../../logs/triplet_loss_features_test.log


