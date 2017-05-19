import caffe
import numpy as np


class MVNLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("MVN layer implemented for only one input")

    def reshape(self, bottom, top):
        # loss output has the same shape as the input
        top[0].reshape(*bottom[0].shape)
        # get other parameters
        self.batch_size = len(bottom[0].data)
        self.norm = np.zeros(self.batch_size)
        self.squares = np.zeros((self.batch_size, len(bottom[0].data[0])))

    def forward(self, bottom, top):
        for i in range(self.batch_size):
            self.squares[i] = bottom[0].data[i]**2
            self.norm[i] = np.sqrt(np.sum(self.squares[i]))
            #print self.norm[i]
            top[0].data[i] = bottom[0].data[i] / self.norm[i]
	    #print np.linalg.norm(top[0].data[i])

    def backward(self, top, propagate_down, bottom):
        if propagate_down:
            for i in range(self.batch_size):
                temp = self.squares[i] - bottom[0].data[i]**2
                my_diff = temp / self.squares[i]**2.5
                bottom[0].diff[i] = top[0].diff[i] * my_diff

