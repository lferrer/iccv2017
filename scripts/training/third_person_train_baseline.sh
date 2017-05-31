#!/usr/bin/env sh
../../caffe/build/install/bin/caffe \
train \
-solver=../../models/third_person_solver_baseline.prototxt --gpu=7 \
-weights=../../../c3d_ucf101_iter_38000.caffemodel \
2>&1 | tee ../../logs/third_person_scene_baseline_train.log
