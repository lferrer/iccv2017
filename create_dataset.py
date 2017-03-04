import string
from os import path
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
INTER_VIDEO_RATE = 0.080
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
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
INTRA_VIDEO_SSIM_THRESHOLD = 0.5
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
    image_name = str(index) + '.jpg'
    if first_person:
        filename = path.join(DS_ROOT, 'First Person', scenario, image_name)
    else:
        angle_folder = str(elevation) + '-' + str(rotation)
        filename = path.join(DS_ROOT, 'Third Person', scenario, angle_folder, image_name)
    return filename

# Returns a random element from a list
def get_rnd_el(my_list):
    return my_list[randint(0, len(my_list) - 1)]

# Returns a random negative inter-video image tuple 
def get_rnd_neg_inter_img(first_person, ref_scenario):
    scenario = get_rnd_el(SCENARIOS)
    while scenario == ref_scenario:
        scenario = get_rnd_el(SCENARIOS)
    index = randint(1, VIDEO_LENGTH - CLIP_LENGTH - 1)
    if first_person:
        neg_filename = build_filename(True, scenario, index)
    else:
        elevation = get_rnd_el(ELEVATION_ANGLES)
        rotation = get_rnd_el(ROTATION_ANGLES)
        neg_filename = build_filename(False, scenario, index, elevation, rotation)
    return neg_filename

# Returns a random negative intra-video image tuple
def get_rnd_neg_intra_img(first_person, scenario, ref_index):
    delta = randint(VIDEO_LENGTH / 2, VIDEO_LENGTH - CLIP_LENGTH - 1)
    index = ref_index + delta
   
    if index >= VIDEO_LENGTH - CLIP_LENGTH:
        index = index - VIDEO_LENGTH - CLIP_LENGTH
    if index < 0:
        index = VIDEO_LENGTH - CLIP_LENGTH - index
    index = max(index, 1)
    
    if first_person:
        neg_filename = build_filename(True, scenario, index)
    else:
        elevation = get_rnd_el(ELEVATION_ANGLES)
        rotation = get_rnd_el(ROTATION_ANGLES)
        neg_filename = build_filename(False, scenario, index, elevation, rotation)
    return neg_filename

def get_base_filename(filename):
    sub_filename = filename[-10:]
    slash_index = string.find(sub_filename, '/')
    index = int(sub_filename[slash_index + 1:-4]) # all files end in .jpg
    image_file_length = 5 - slash_index + 4
    base_filename = filename[:-image_file_length]
    return base_filename, index

def get_random_triplet(fp_anchor):
    scenario = get_rnd_el(SCENARIOS)
    index = randint(1, VIDEO_LENGTH - CLIP_LENGTH - 1)
    fp_filename = build_filename(True, scenario, index)
    elevation = get_rnd_el(ELEVATION_ANGLES)
    rotation = get_rnd_el(ROTATION_ANGLES)
    tp_filename = build_filename(False, scenario, index, elevation, rotation)

    # Choose if the negative sample is going to be a inter or intra-video sample
    if random() < INTER_VIDEO_RATE:
        negative_filename = get_rnd_neg_inter_img(not fp_anchor, scenario)
    else:
        negative_filename = get_rnd_neg_intra_img(not fp_anchor, scenario, index)

    # Choose the anchor
    if fp_anchor:
        anchor_filename = fp_filename
        positive_filename = tp_filename
    else:
        anchor_filename = tp_filename
        positive_filename = fp_filename

    return anchor_filename, positive_filename, negative_filename

def add_to_mean(image):
    global image_mean, image_count
    mean = np.mean(np.mean(image, axis=0), axis=0)
    image_mean = image_mean * image_count + mean
    image_count = image_count + 1
    image_mean = image_mean / image_count


def load_image_triplet(anchor_filename, positive_filename, negative_filename):
    # Retrieve the index number
    base_anchor_filename, anchor_index = get_base_filename(anchor_filename)
    base_positive_filename, positive_index = get_base_filename(positive_filename)
    base_negative_filename, negative_index = get_base_filename(negative_filename)
    
    # Load the base images from disk
    anchor_image = cv2.imread(anchor_filename)
    positive_image = cv2.imread(positive_filename)
    negative_image = cv2.imread(negative_filename)

    # INTER_AREA is the best for image decimation (According to OpenCV documentation)
    anchor_image_data = cv2.resize(anchor_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)
    positive_image_data = cv2.resize(positive_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)
    negative_image_data = cv2.resize(negative_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)

    # Add up to the mean calculation
    add_to_mean(anchor_image_data)
    add_to_mean(positive_image_data)
    add_to_mean(negative_image_data)

    # Load an stack the rest on them
    for i in range(1, CLIP_LENGTH):
        #Update the filenames
        anchor_filename = base_anchor_filename + str(anchor_index + i) + '.jpg'
        positive_filename = base_positive_filename + str(positive_index + i) + '.jpg'
        negative_filename = base_negative_filename + str(negative_index + i) + '.jpg'

        # Load the images from disk
        anchor_image = cv2.imread(anchor_filename)
        positive_image = cv2.imread(positive_filename)
        negative_image = cv2.imread(negative_filename)

        # INTER_AREA is the best for image decimation (According to OpenCV documentation)
        anchor_image = cv2.resize(anchor_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)
        positive_image = cv2.resize(positive_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)
        negative_image = cv2.resize(negative_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)

        # Add up to the mean calculation
        add_to_mean(anchor_image)
        add_to_mean(positive_image)
        add_to_mean(negative_image)

        # Stack the images for datum insertion
        anchor_image_data = np.dstack((anchor_image, anchor_image_data))
        positive_image_data = np.dstack((positive_image, positive_image_data))
        negative_image_data = np.dstack((negative_image, negative_image_data))

    image_data = np.dstack((anchor_image_data, positive_image_data, negative_image_data))
    return image_data

def add_to_db(txn, image_data, i):
    # Build the caffe Datum
    datum = caffe.proto.caffe_pb2.Datum()
    datum.height = image_data.shape[0]
    datum.width = image_data.shape[1]
    datum.channels = image_data.shape[2]
    datum.data = image_data.tobytes()
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

def generate_database(db_name, n_samples, index_start):
    n_batches = int(math.ceil(n_samples / SAMPLE_BATCH_RATE))
    index = 0
    fp_anchor = True
    for batch in range(n_batches):
        print "Creating batch {} out of {}".format(batch + 1, n_batches)
        env = lmdb.open(db_name, map_size=MAP_SIZE)
        with env.begin(write=True) as txn:
            n_samples_left = int(min(SAMPLE_BATCH_RATE, n_samples - SAMPLE_BATCH_RATE*batch))
            for i in range(n_samples_left):

                # Get a random triplet
                anchor_filename, positive_filename, negative_filename = get_random_triplet(fp_anchor)
                img_triplet = "{}-{}-{}".format(anchor_filename, positive_filename, negative_filename)

                while img_triplet in hit_dict:
                    # Get an unused random triplet
                    anchor_filename, positive_filename, negative_filename = get_random_triplet(fp_anchor)
                    img_triplet = "{}-{}-{}".format(anchor_filename, positive_filename, negative_filename)

                # Add the random image to the used list
                hit_dict[img_triplet] = True

                # Get the images from the disk
                image_data = load_image_triplet(anchor_filename, positive_filename, negative_filename)

                # Add to the lmdb
                db_index = int(index_list[index + index_start + i])
                add_to_db(txn, image_data, db_index)

                # Invert the anchor to generate another type of sample
                fp_anchor = not fp_anchor

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

# Create a random index list to shuffle the data
index_list = create_index_list(n_samples)

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
