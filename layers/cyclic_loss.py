import caffe
import numpy as np


class CyclicLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need four inputs to compute cyclic loss.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs 0 and 1 must have the same dimension.")
        if bottom[0].count != bottom[2].count:
            raise Exception("Inputs 0 and 2 must have the same dimension.")
        if bottom[0].count != bottom[3].count:
            raise Exception("Inputs 0 and 3 must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
        # get other parameters
        self.batch_size = len(bottom[0].data)
        self.first_person = True

    def forward(self, bottom, top):
        if self.first_person:
            diff_F = bottom[1].data - bottom[2].data
            diff_G = bottom[0].data - bottom[3].data
        else:
            diff_F = bottom[0].data - bottom[3].data
            diff_G = bottom[1].data - bottom[2].data
        # Storing for backward pass
        self.diff[...] = diff_F + diff_G
        self.first_person = False

        top[0].data[...] = np.sum(diff_F**2) / bottom[1].num / 2. + np.sum(diff_G**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(4):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            elif i == 1:
                sign = 1
            elif i == 2:
                sign = -1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num

