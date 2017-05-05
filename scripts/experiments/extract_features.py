import sys
import os
import numpy as np
import caffe
import lmdb

CLIP_LENGTH = 16

def read_lmdb(lmdb_file):
    cursor = lmdb.open(lmdb_file, readonly=True).begin().cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    for _, value in cursor:
        datum.ParseFromString(value)
        yield datum.data, datum.label

if len(sys.argv) < 7:
    print "Error: Not enough parameters given. Parameters needed: "
    print "-gpu -weights_file -model_file -features_file -lmdb_folder -batch_size"
    exit()
else:
    GPU_ID = int(sys.argv[1])
    WEIGHTS_FILE = sys.argv[2]
    model_file = sys.argv[3]
    features_file = sys.argv[4]
    lmdb_folder = sys.argv[5]
    BATCH_SIZE = int(sys.argv[6])

# Start Caffe
caffe.set_device(GPU_ID)
caffe.set_mode_gpu()
net = caffe.Net(MODEL, 1, weights=WEIGHTS_FILE)

# Get a pointer to the file
filenames_file = os.path.join(lmdb_folder, "image_list.csv")
image_filenames = [line.rstrip('\n') for line in open(filenames_file, "r")]

# Calculate the number of batches
n_batches = len(image_filenames) / BATCH_SIZE
filename_index = 0

# Allocate the space to store the features
features = np.empty([len(image_filenames), 2048])
features_index = 0

# Preallocate space
image_data = np.empty([BATCH_SIZE,
                       3,         # 3-channel (BGR) images
                       CLIP_LENGTH,
                       CROP_WIDTH,
                       CROP_HEIGHT], dtype=np.uint8)

# Create batches
n_images_in_batch = 0
for im, label in read_lmdb(lmdb_folder):
    if n_images_in_batch == BATCH_SIZE:
        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = image_data
        output = net.forward()
        output_features = output['fc7']
        for feature in output_features:
            features[features_index] = feature
            features_index = features_index + 1

for batch_index in range(n_batches):
    
    # Generate a batch
    for j in range(BATCH_SIZE):
        image_filename = DS_ROOT + image_filenames[filename_index]
        sample_image_data, sample_index = load_image_sample(image_filename)
        filename_index = filename_index + 1

        # Stack the images over the depth dimension
        image_data[j] = sample_image_data

    

    print "Batch {}/{} done".format(batch_index + 1, n_batches)

np.save(features_file, features)


