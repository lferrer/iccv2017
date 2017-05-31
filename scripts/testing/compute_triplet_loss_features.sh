#!/usr/bin/env sh
../../caffe/build/install/bin/caffe \
test \
-model=../../models/three_stream_deploy.prototxt -gpu=3 \
-weights=../../weights/three_stream_triplet_loss_previous_iter_60000.caffemodel \
-iterations=1000 \
2>&1 | tee ../../logs/triplet_loss_features_previous_train.log


