#NOTE: REQUIRES OpenCV 3+
import string
import os
import sys
import math
from random import randint, random

import cv2
import lmdb
import numpy as np
import caffe

# Config section of the script
TRAINING_RATE = 0.7
VALIDATION_RATE = 0.1
TESTING_RATE = 0.2
SAMPLE_PRINT_RATE = 100.0
SAMPLE_BATCH_RATE = 1000.0
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

#According to the LMDB website: On 64-bit there is no penalty for making this huge (say 1TB)
MAP_SIZE = 1e12

# Variables
hit_dict = {}
image_count = 0
image_mean = np.zeros((1, 3))

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

def add_to_mean(image):
    global image_mean, image_count
    mean = np.mean(np.mean(image, axis=0), axis=0)
    image_mean = image_mean * image_count + mean
    image_count = image_count + 1
    image_mean = image_mean / image_count


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

    # Add up to the mean calculation
    add_to_mean(sample_image_data)

    # Load an stack the rest on them
    for i in range(1, CLIP_LENGTH):
        #Update the filenames
        sample_filename = base_sample_filename + str(sample_index + i) + '.jpg'

        # Load the images from disk
        sample_image = cv2.imread(sample_filename)

        # INTER_AREA is the best for image decimation (According to OpenCV documentation)
        sample_image = cv2.resize(sample_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)

        # Add up to the mean calculation
        add_to_mean(sample_image)

        # Stack the images for datum insertion
        sample_image_data = np.dstack((sample_image, sample_image_data))

    return sample_image_data, sample_index

def add_to_db(txn, image_data, label, i):
    # Build the caffe Datum
    datum = caffe.proto.caffe_pb2.Datum()
    datum.height = image_data.shape[0]
    datum.width = image_data.shape[1]
    datum.channels = image_data.shape[2]
    datum.data = image_data.tobytes()
    datum.label = label
    str_id = '{:08}'.format(i)

    # The encode is only essential in Python 3
    txn.put(str_id.encode('ascii'), datum.SerializeToString())

def create_index_list(n_samples):
    index_list = np.zeros(n_samples)

    for i in range(n_samples):
        pos = randint(0, n_samples - 1)
        while index_list[pos] != 0:
            pos = randint(0, n_samples - 1)
        index_list[pos] = i

    return index_list

def get_label(index):
    global anim_durs
    for i in range(len(anim_durs)):
        if anim_durs[i] <= index:
            return i
    return -1 #Error case handled gracefully

def generate_database(db_name, n_samples, index_start):
    n_batches = int(math.ceil(n_samples / SAMPLE_BATCH_RATE))
    index = 0
    fp_sample = True
    for batch in range(n_batches):
        print "Creating batch {} out of {}".format(batch + 1, n_batches)
        env = lmdb.open(db_name, map_size=MAP_SIZE)
        with env.begin(write=True) as txn:
            n_samples_left = int(min(SAMPLE_BATCH_RATE, n_samples - SAMPLE_BATCH_RATE*batch))
            for i in range(n_samples_left):
                # Get a random sample
                sample_filename = get_random_sample(fp_sample)
                image_data, image_index = load_image_sample(sample_filename)

                # Be sure it is an unused sample and is valid
                while sample_filename in hit_dict and image_index < 0:
                    # Get an unused random triplet
                    sample_filename = get_random_sample(fp_sample)
                    # Get the images from the disk
                    image_data, image_index = load_image_sample(sample_filename)

                # Add the random image to the used list
                hit_dict[sample_filename] = True

                # Add to the lmdb
                db_index = int(index_list[index + index_start + i])
                image_label = get_label(image_index)
                add_to_db(txn, image_data, image_label, db_index)

                # Invert to generate another type of sample
                fp_sample = not fp_sample

                if (i + 1) % SAMPLE_PRINT_RATE == 0:
                    print "Generated {} out of {} samples".format(index + i + 1, int(n_samples))

        index = index + n_samples_left

# Create DBs
if len(sys.argv) < 2:
    print "Error: Not enough parameters given. Parameters needed: -db_path -n_samples"
    exit()
else:
    db_path = sys.argv[1]
    n_samples = int(sys.argv[2])

print 'Generating database here: {} with {} samples'.format(db_path, n_samples)

if not(os.path.exists(db_path)):
	os.mkdir(db_path)

# Create a random index list to shuffle the data
index_list = create_index_list(n_samples)

# Create a list of indices based on the animations
anim_durs = [88,176,366,556,576,596,655,714,773,832,891,950,1009,1068,1132,1194,1253,1328,1390,1444,1498,1507,1516,1557,1598,1629,1660,1691,1722,1912,2102,2253,2379,2569,2615,2661,2699,2745,2770,2810,2850,2873,2932,2941,2950,3206,3307,3437,3482,3534,3601,3668,3798,3928,3959,3985,4012,4063,4120,4177,4234,4291]

# Generate samples
train_samples = int(n_samples * TRAINING_RATE)
train_name = db_path + "/train"
print 'Generating train database with {} samples'.format(train_samples)
generate_database(train_name, train_samples, 0)

val_samples = int(n_samples * VALIDATION_RATE)
val_name = db_path + "/val"
print 'Generating val database with {} samples'.format(val_samples)
generate_database(val_name, val_samples, train_samples)

test_samples = int(n_samples * TESTING_RATE)
test_name = db_path + "/test"
print 'Generating test database with {} samples'.format(test_samples)
generate_database(test_name, test_samples, train_samples + val_samples)

f = open(db_path + "/image_mean.txt","w+")
f.write(str(image_mean))
f.close()

print 'Generation complete. Image mean: ' + str(image_mean)
exit()
