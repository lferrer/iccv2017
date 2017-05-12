#!/usr/bin/env sh
../../caffe/build/install/bin/caffe \
test \
-model=../../models/three_stream_deploy.prototxt -gpu=7 \
-weights=../../../c3d_ucf101_iter_38000.caffemodel \
-iterations=1000 \
2>&1 | tee ../../logs/baseline_features_test.log

