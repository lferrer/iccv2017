import caffe

class LabelSplitLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need access to the raw_label only.")
        # check output pair
        if len(top) != 2:
            raise Exception("Need two output labels.")
        # Specifying type of input data
        self.first_person = True


    def reshape(self, bottom, top):
        # check input dimensions match
        top[0].reshape(len(bottom[0].data), 1)
        top[1].reshape(len(bottom[0].data), 1)

    def forward(self, bottom, top):
        for i, raw_label in enumerate(bottom[0].data):
            if raw_label < 1000000:
                #6 digits label
                label_str = '{0:06.0f}'.format(raw_label)
                left_label = raw_label[:3]
                right_label = raw_label[3:]
            elif raw_label < 10000000:
                #7 digits label
                label_str = '{0:07.0f}'.format(raw_label)
                if label_str[:2] == '10':
                    # left is longer
                    left_label = raw_label[:4]
                    right_label = raw_label[4:]
                else:
                    # right is longer
                    left_label = raw_label[:3]
                    right_label = raw_label[3:]
            else:
                # 8 digits label
                label_str = '{0:08.0f}'.format(raw_label)
                left_label = raw_label[:4]
                right_label = raw_label[4:]
            if self.first_person:
                top[0].data[i] = int(left_label)
                top[1].data[i] = int(right_label)
            else:
                top[0].data[i] = int(right_label)
                top[1].data[i] = int(left_label)

    def backward(self, top, propagate_down, bottom):
        pass
