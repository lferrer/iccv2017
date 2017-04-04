#!/usr/bin/env sh
./C3D-v1.0/build/tools/finetune_net.bin \
solver.prototxt \
../conv3d_deepnetA_sport1m_iter_1900000 \
2>&1 | tee mini_test.log