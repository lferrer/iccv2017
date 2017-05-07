#!/usr/bin/env sh
../../caffe/build/install/bin/caffe \
test \
-model=../../models/three_stream_deploy.prototxt -gpu=7 \
-weights=../../../c3d_ucf101_iter_38000.caffemodel \
-iterations=1500 \
2>&1 | tee ../../logs/three_stream_triplet_loss_features_test.log

