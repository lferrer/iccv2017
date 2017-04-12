import string
import os
import sys
import math
from random import randint, random

import cv2
import numpy as np
import caffe

CLIP_LENGTH = 16
DS_ROOT = '/data/leo-data/Synthetic/NN/'
# Video frames were to 128 by 171 (According to original paper)
FRAME_WIDTH = 128
FRAME_HEIGHT = 171
MODEL = '../models/three_stream_deploy.prototxt'
WEIGHTS = '../weights/c3d_ucf101_iter_1000.caffemodel'
BATCH_SIZE = 3

# Helper functions
def get_base_filename(filename):
    sub_filename = filename[-10:]
    slash_index = string.find(sub_filename, '/')
    index = int(sub_filename[slash_index + 1:-4]) # all files end in .jpg
    image_file_length = 5 - slash_index + 4
    base_filename = filename[:-image_file_length]
    return base_filename, index

def load_image_sample(sample_filename):
    # Retrieve the index number
    base_sample_filename, sample_index = get_base_filename(sample_filename)

    # Load the base images from disk
    sample_image = cv2.imread(sample_filename)
    height, width = sample_image.shape[:2]
    # Return -1 to flag an invalid image
    if width == 0 or height == 0:
        return sample_image, -1

    # INTER_AREA is the best for image decimation (According to OpenCV documentation)
    sample_image_data = cv2.resize(sample_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)

    # Load an stack the rest on them
    for i in range(1, CLIP_LENGTH):
        #Update the filenames
        sample_filename = base_sample_filename + str(sample_index + i) + '.jpg'

        # Load the images from disk
        sample_image = cv2.imread(sample_filename)

        # INTER_AREA is the best for image decimation (According to OpenCV documentation)
        sample_image = cv2.resize(sample_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)

        # Stack the images for datum insertion
        sample_image_data = np.dstack((sample_image, sample_image_data))

    return sample_image_data, sample_index

if len(sys.argv) < 2:
    print "Error: Not enough parameters given. Parameters needed: -filenames_file -features_path"
    exit()
else:
    filenames_file = sys.argv[1]
    features_path = sys.argv[2]

# Start Caffe
caffe.set_mode_cpu()
net = caffe.Net(MODEL, WEIGHTS, caffe.TEST)

# build the mean object
mu = [('B', 67.1991691), ('G', 75.33919712), ('R', 90.82602441)]

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(BATCH_SIZE,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

# Get a pointer to the file
image_filenames = [line.rstrip('\n') for line in open(filenames_file, "r")]
features_file = open(features_path, "w")

for index in range(len(image_filenames)):
    image_data = np.empty([FRAME_HEIGHT, FRAME_WIDTH, 3*CLIP_LENGTH], dtype=np.uint8)
    image_filename = DS_ROOT + image_filenames[index]
    # Generate a batch
    for j in range(BATCH_SIZE):
        sample_image_data, sample_index = load_image_sample(image_filename)
        while sample_index == -1:
            index = index + 1
            image_filename = DS_ROOT + image_filenames[index]
            sample_image_data, sample_index = load_image_sample(image_filename)

        # Stack the images over the depth dimension
#        print image_data.shape, sample_image_data.shape
        image_data = np.dstack((sample_image_data, image_data))

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = image_data
    output = net.forward()
    features_file.write(output + "\n")

list_file.close()
features_file.close()


