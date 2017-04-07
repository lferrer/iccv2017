import string
import os
import sys
import math
from random import randint, random

import cv2
import numpy as np
import caffe

CLIP_LENGTH = 16
VIDEO_LENGTH = 4292
SCENARIOS = ['Fire Pit', 'Grass Standard', 'Ice Natural', 'Reflection Agent', 'Temple Barbarous',
             'Fire Natural', 'Grass Pit', 'Ice Barbarous', 'Reflection Standard', 'Temple Agent']
ELEVATION_ANGLES = [330, 340, 350]
ROTATION_ANGLES = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, \
                   200, 220, 240, 260, 280, 300, 320, 340]
DS_ROOT = '/home/lferrer/Documents/Synthetic'
# Video frames were to 128 by 171 (According to original paper)
FRAME_WIDTH = 128
FRAME_HEIGHT = 171
MODEL = '../models/three_stream_deploy.prototxt'
WEIGHTS = '../weights/c3d_ucf101_iter_1000.caffemodel'
BATCH_SIZE = 48
TEST_ITERATIONS = 10

# Variables
hit_dict = {}

# Helper functions
# Returns the filename of a given image based on the dataset structure
def build_filename(first_person, scenario, index, elevation=330, rotation=0):
    if first_person:
        # First-Person goes from 1 - 4192
        image_name = str(index + 1) + '.jpg'
        filename = os.path.join(DS_ROOT, 'First Person', scenario, image_name)
    else:
        # Third person goes from 0 to 4191
        image_name = str(index) + '.jpg'
        angle_folder = str(elevation) + '-' + str(rotation)
        filename = os.path.join(DS_ROOT, 'Third Person', scenario, angle_folder, image_name)
    return filename

# Returns a random element from a list
def get_rnd_el(my_list):
    return my_list[randint(0, len(my_list) - 1)]

def get_base_filename(filename):
    sub_filename = filename[-10:]
    slash_index = string.find(sub_filename, '/')
    index = int(sub_filename[slash_index + 1:-4]) # all files end in .jpg
    image_file_length = 5 - slash_index + 4
    base_filename = filename[:-image_file_length]
    return base_filename, index

def get_random_sample(fp_sample):
    scenario = get_rnd_el(SCENARIOS)
    index = randint(1, VIDEO_LENGTH - CLIP_LENGTH - 1)
    if fp_sample:
        return build_filename(True, scenario, index)
    else:
        elevation = get_rnd_el(ELEVATION_ANGLES)
        rotation = get_rnd_el(ROTATION_ANGLES)
        return build_filename(False, scenario, index, elevation, rotation)

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
fp_sample = True
for i in range(TEST_ITERATIONS):
    image_data = np.array([], dtype=np.uint8).reshape(3, FRAME_WIDTH, FRAME_HEIGHT)

    # Generate a batch
    for j in range(BATCH_SIZE):
        sample_filename = get_random_sample(fp_sample)
        sample_image_data, sample_index = load_image_sample(sample_filename)
        while sample_filename in hit_dict and sample_index == -1:
            sample_filename = get_random_sample(fp_sample)
            sample_image_data, sample_index = load_image_sample(sample_filename)

        # Add the random image to the used list
        hit_dict[sample_filename] = True

        # Stack the images over the depth dimension
        image_data = np.dstack((sample_image_data, image_data))

        # Invert to generate another type of sample
        fp_sample = not fp_sample

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = image_data
    output = net.forward()
    


