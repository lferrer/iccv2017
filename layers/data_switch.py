import caffe
import numpy as np


class DataSwitchLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute cyclic loss.")
        # check output pair
        if len(top) != 2:
            raise Exception("Need two inputs to compute cyclic loss.")
        # Specifying type of input data
        self.first_person = True


    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs 0 and 1 must have the same dimension.")

    def forward(self, bottom, top):
        if self.first_person:
            top[0].data[...] = bottom[0].data[...]
            top[1].data[...] = bottom[1].data[...]
        else:
            top[1].data[...] = bottom[0].data[...]
            top[0].data[...] = bottom[1].data[...]

    def backward(self, top, propagate_down, bottom):
        pass
