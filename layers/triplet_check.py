import caffe
import numpy as np
import json

class TripletCheckLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three inputs to check a triplet")
        if hasattr(self, 'param_str') and self.param_str:
            params = json.loads(self.param_str)            
            self.n_samples = int(params['n_samples'])
            self.filename = params['filename']
        else:
            raise Exception("Need to setup param_str")
        self.sample_index = 0
        self.n_epochs = 0
    def reshape(self, bottom, top):
         # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs 0 and 1 must have the same dimension.")
        if bottom[0].count != bottom[2].count:
            raise Exception("Inputs 0 and 2 must have the same dimension.")
        self.batch_size = len(bottom[0].data)
        if not hasattr(self, 'distances'):
            self.distances = np.empty([self.n_samples,
                                      len(bottom[0].data[0])],
                                     dtype=float)
    def forward(self, bottom, top):
        for i in range(self.batch_size):
            dist_same_feature = np.linalg.norm(bottom[0].data[i] - bottom[1].data[i])**2
            dist_diff_feature = np.linalg.norm(bottom[0].data[i] - bottom[2].data[i])**2
	    if self.sample_index + i < self.n_samples:
                self.distances[self.sample_index + i] = dist_same_feature + dist_diff_feature

        self.sample_index = self.sample_index + self.batch_size
        if self.sample_index >= self.n_samples:
            self.n_epochs += 1
            self.sample_index = 0
            my_filename = self.filename + str(self.n_epochs) + '.npz'
            np.savez_compressed(my_filename, self.distances)
            print 'Features saved to: ' + my_filename

    def backward(self, top, propagate_down, bottom):
        pass
