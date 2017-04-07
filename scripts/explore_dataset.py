import caffe
import lmdb
import PIL.Image
from StringIO import StringIO
import numpy as np

def read_lmdb(lmdb_file):
    cursor = lmdb.open(lmdb_file, readonly=True).begin().cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    for key, value in cursor:
        datum.ParseFromString(value)
        #s = StringIO()
        #s.write(datum.data)
        #s.seek(0)

        #yield np.array(PIL.Image.open(s)), datum.label, key
        yield key, datum.label


lmdb_dir = 'train'
for key, label in read_lmdb(lmdb_dir):
    print key, label