from StringIO import StringIO
import lmdb
import caffe
import numpy as np
import sys

def read_lmdb(lmdb_file):
    cursor = lmdb.open(lmdb_file, readonly=True).begin().cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    for _, value in cursor:
        datum.ParseFromString(value)
        s = StringIO()
        s.write(datum.data)
        s.seek(0)

        yield np.array(s), datum.label

if len(sys.argv) < 4:
    print "Error: Not enough parameters given. Parameters needed: -gpu -lmdb_dir -model"
    exit()
else:
    gpu_id = int(sys.argv[1])
    lmdb_dir = sys.argv[2]
    model_file = sys.argv[3]

for im, label in read_lmdb(lmdb_dir):
    print label


