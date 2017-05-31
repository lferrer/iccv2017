#!/usr/bin/env sh
../../caffe/build/install/bin/caffe \
test \
-model=../../models/first_person_deploy.prototxt -gpu=1 \
-weights=../../../third_person_scene_iter_18000.caffemodel \
-iterations=1000 \
2>&1 | tee ../../logs/first_person.log

