import string
import sys
import os
from random import randint

import cv2
import numpy as np
np.set_printoptions(threshold=np.inf) #Pretty print vectors
import caffe

CLIP_LENGTH = 16
DS_ROOT = '/data/leo-data/Synthetic/NN/'
#DS_ROOT = '/home/lferrer/Documents/Synthetic/NN/'
# Video frames were to scaled to 128 by 171 (According to original paper)
FRAME_WIDTH = 128
FRAME_HEIGHT = 171
# And then cropped to 112 by 112
CROP_WIDTH = 112
CROP_HEIGHT = 112
MODEL = '../models/three_stream_deploy.prototxt'
WEIGHTS = '../weights/c3d_ucf101_iter_1000.caffemodel'
BATCH_SIZE = 18

# Helper functions
def get_base_filename(filename):
    sub_filename = filename[-10:]
    slash_index = string.find(sub_filename, '/')
    index = int(sub_filename[slash_index + 1:-4]) # all files end in .jpg
    image_file_length = 5 - slash_index + 4
    base_filename = filename[:-image_file_length]
    return base_filename, index

def get_random_crop(sample_image):
    x_coord = randint(0, FRAME_WIDTH - CROP_WIDTH)
    y_coord = randint(0, FRAME_HEIGHT - CROP_HEIGHT)
    return sample_image[y_coord:y_coord + CROP_HEIGHT, x_coord:x_coord + CROP_WIDTH, :]

def load_image_sample(sample_filename):
    # Check if the image even exists
    if not os.path.exists(sample_filename):
        return [], -1

    # Retrieve the index number
    base_sample_filename, sample_index = get_base_filename(sample_filename)

    sample_image_data = np.empty([3,         # 3-channel (BGR) images
                                  CLIP_LENGTH,
                                  CROP_WIDTH,
                                  CROP_HEIGHT], dtype=np.uint8)

    # Load an stack the rest on them
    for i in range(CLIP_LENGTH):
        #Update the filenames
        sample_filename = base_sample_filename + str(sample_index + i) + '.jpg'

        # Load the images from disk
        sample_image = cv2.imread(sample_filename)

        # INTER_AREA is the best for image decimation (According to OpenCV documentation)
        sample_image = cv2.resize(sample_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)

        # Get a random crop
        sample_image = get_random_crop(sample_image)

        # Subtract the mean
        sample_image = sample_image - MU

        # Stack the images for datum insertion
        for c in range(3):
            sample_image_data[c, i] = sample_image[:, :, c]

    return sample_image_data, sample_index

if len(sys.argv) < 2:
    print "Error: Not enough parameters given. Parameters needed: -filenames_file -features_path"
    exit()
else:
    filenames_file = sys.argv[1]
    features_path = sys.argv[2]

# Start Caffe
caffe.set_device(1)
caffe.set_mode_gpu()
net = caffe.Net(MODEL, 1, weights=WEIGHTS)
#net = caffe.Net(MODEL, 1)

# build the mean object
MU = [67, 75, 91]

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(BATCH_SIZE,
                          3,         # 3-channel (BGR) images
                          CLIP_LENGTH,
                          CROP_WIDTH,
                          CROP_HEIGHT)

# Get a pointer to the file
image_filenames = [line.rstrip('\n') for line in open(filenames_file, "r")]
features_file = open(features_path, "w")

for index in enumerate(image_filenames):
    image_data = np.empty([BATCH_SIZE,
                           3,         # 3-channel (BGR) images
                           CLIP_LENGTH,
                           CROP_WIDTH,
                           CROP_HEIGHT], dtype=np.uint8)
    image_filename = DS_ROOT + image_filenames[index]
    # Generate a batch
    for j in range(BATCH_SIZE):
        sample_image_data, sample_index = load_image_sample(image_filename)
        while sample_index == -1:
            index = index + 1
            image_filename = DS_ROOT + image_filenames[index]
            sample_image_data, sample_index = load_image_sample(image_filename)

        # Stack the images over the depth dimension
        image_data[j] = sample_image_data

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = image_data
    output = net.forward()
    output_features = output['fc7']
    for feature in output_features:
        np.save(features_file, feature)

features_file.close()
