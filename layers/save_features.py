import caffe
import numpy as np


class SaveFeaturesLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) == 0:
            raise Exception("Need at least one input")
        if hasattr(self, 'param_str') and self.param_str:
            self.n_samples = int(self.param_str[0])
        else:
            raise Exception("Need to setup param_str")
        self.sample_index = 0
    def reshape(self, bottom, top):
        self.n_inputs = len(bottom)
        self.batch_size = len(bottom[0].data)
        self.features = np.empty([self.n_inputs,
                                  self.n_samples,
                                  len(bottom[0].data[0])],
                                 dtype=float)

    def forward(self, bottom, top):
        for i in range(self.n_inputs):
            for j in range(self.batch_size):
                self.features[i][self.sample_index + j] = bottom[i].data[j]
        self.sample_index = self.sample_index + self.batch_size
        if self.sample_index == self.n_samples:
            np.savez_compressed('features.npz', self.features)

    def backward(self, top, propagate_down, bottom):
        pass