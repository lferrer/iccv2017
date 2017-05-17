import caffe
import numpy as np


class MVNLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("MVN layer implemented for only one input")

    def reshape(self, bottom, top):
        # loss output has the same shape as the input
        top[0].reshape(bottom[0].shape)
        # get other parameters
        self.batch_size = len(bottom[0].data)
        self.norm = np.zeros(self.batch_size)

    def forward(self, bottom, top):
        for i in range(self.batch_size):
            self.norm[i] = np.linalg.norm(bottom[0].data[i])**2
            top[0].data[i] = bottom[0].data[i] / self.norm[i]

    def backward(self, top, propagate_down, bottom):
        if propagate_down:
            for i in range(self.batch_size):
                bottom[0].diff[i] = top[0].diff[i] * self.norm[i]
