#!/usr/bin/env sh
../../caffe/build/install/bin/caffe \
train \
-solver=../../models/third_person_solver.prototxt --gpu=3 \
-weights=../../weights/three_stream_triplet_loss_iter_60000.caffemodel \
2>&1 | tee ../../logs/third_person_scene_train.log
