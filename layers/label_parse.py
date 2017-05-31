import caffe

class LabelParseLayer(caffe.Layer):
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
        top[0].reshape(len(bottom[0].data), 1)

    def forward(self, bottom, top):
        for i, label in enumerate(bottom[0].data):
            label = str(int(label[0]))
            if self.l_type:
                my_value = label[:len(label) - 2]
            else:
                my_value = label[-2:]
            if len(my_value) == 0:
                my_value = '11'
            top[0].data[i] = int(my_value)

    def backward(self, top, propagate_down, bottom):
        pass
