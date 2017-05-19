import caffe
import json
import numpy as np

class SaveFeaturesLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) == 0:
            raise Exception("Need at least one input")
        if hasattr(self, 'param_str') and self.param_str:
            params = json.loads(self.param_str)
            self.n_samples = int(params['n_samples'])
            self.filename = params['filename']
        else:
            raise Exception("Need to setup param_str")
        self.sample_index = 0
    def reshape(self, bottom, top):
        self.n_inputs = len(bottom)
        self.batch_size = len(bottom[0].data)
        if not hasattr(self, 'features'):
            self.features = np.empty([self.n_inputs,
                                      self.n_samples,
                                      len(bottom[0].data[0])],
                                     dtype=float)
        #print "2132342134244234"

    def forward(self, bottom, top):
        for i in range(self.n_inputs):
            for j in range(self.batch_size):
                self.features[i][self.sample_index + j] = bottom[i].data[j]
                #print bottom[i].data[j]
                #print self.features[i][self.sample_index + j]
        #print np.nonzero(self.features[0][0])
        self.sample_index = self.sample_index + self.batch_size
        if self.sample_index == self.n_samples:
            np.savez_compressed(self.filename, self.features)
	    print 'Features saved to: ' + self.filename

    def backward(self, top, propagate_down, bottom):
        pass

