import caffe
import numpy as np

class LabelSplitLayer(caffe.Layer):
    def setup(self, bottom, top):
        if hasattr(self, 'param_str') and self.param_str:
            if self.param_str == 'Scene':
                self.l_type = True
            elif self.param_str == 'Action':
                self.l_type = False
            else:
                raise Exception("Unsupported label type: " + self.param_str)
        else:
            raise Exception("Need to setup param_str")

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for i, label in enumerate(bottom[0].data):
            if self.l_type:
                #top[0].data[i] = label[:len(label) - 2]
                if len(label) == 4:
                    top[0].data[i] = np.asarray(label[:2])
                else:
                    top[0].data[i] = np.asarray(label[:1])
            else:
                top[0].data[i] = np.asarray(label[-2:])

    def backward(self, top, propagate_down, bottom):
        pass
