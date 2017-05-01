import caffe
import numpy as np


class TripletLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute triplet loss.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs 0 and 1 must have the same dimension.")
        if bottom[0].count != bottom[2].count:
            raise Exception("Inputs 0 and 2 must have the same dimension.")
        # difference is shape of inputs
        self.diff_same_class = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff_diff_class = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
        # get other parameters
        self.batch_size = len(bottom[0].data)
        self.vec_dimension = len(bottom[0].data[0])
        self.vec_loss = np.zeros(self.batch_size, dtype=np.float32)

    def forward(self, bottom, top):
        self.diff_same_class[...] = bottom[0].data - bottom[1].data
        self.diff_diff_class[...] = bottom[0].data - bottom[2].data

        loss = 0
        ALPHA = 1.0
        for v in range(self.batch_size):
            self.vec_loss[v] = ALPHA + \
                               np.linalg.norm(self.diff_same_class[v])**2 - \
                               np.linalg.norm(self.diff_diff_class[v])**2
            self.vec_loss[v] = max(0, self.vec_loss[v])
            loss = loss + self.vec_loss[v]
        top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        for i in range(3):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 2
                id_1 = 2 # Negative
                id_2 = 1 # Positive
            elif i == 1:
                sign = -2
                id_1 = 0 # Anchor
                id_2 = 1 # Positive
            else:
                sign = 2
                id_1 = 0 # Anchor
                id_2 = 2 # Negative
            bottom[i].diff[...] = sign * (bottom[id_1].data - bottom[id_2].data)
            for v in range(self.batch_size):
                if self.vec_loss[v] == 0:
                    bottom[i].diff[v] = np.zeros_like(self.vec_dimension, dtype=np.float32)
