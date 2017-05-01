import numpy as np
import caffe

MODEL = '../models/dummy_model.prototxt'
net = caffe.Net(MODEL, 1)

# Reshaping to test 2 vectors only
net.blobs['anchor'].reshape(2, 2048)
net.blobs['positive'].reshape(2, 2048)
net.blobs['negative'].reshape(2, 2048)

# Inputing the vectors
net.blobs['anchor'].data[...] = [np.full(2048, 1, dtype=np.float32), np.full(2048, 2, dtype=np.float32)]
net.blobs['positive'].data[...] = [np.full(2048, 2, dtype=np.float32), np.full(2048, 8, dtype=np.float32)]
net.blobs['negative'].data[...] = [np.full(2048, 1, dtype=np.float32), np.full(2048, 4, dtype=np.float32)]

# Test the forward pass
forward = net.forward()
output_features = forward['loss']
backward = net.backward()
