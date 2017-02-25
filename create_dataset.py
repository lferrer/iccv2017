import string
from os import path
from random import randint

import cv2
import lmdb
import numpy as np
from skimage.measure import compare_ssim as ssim

import caffe

# Config section of the script
POSITIVE_TRAIN_SAMPLES = 7#140000
NEGATIVE_TRAIN_SAMPLES = 7#140000
POSITIVE_VAL_SAMPLES = 1#20000
NEGATIVE_VAL_SAMPLES = 1#20000
POSITIVE_TEST_SAMPLES = 2#40000
NEGATIVE_TEST_SAMPLES = 2#40000
INTER_VIDEO_RATE = 0.80

# 4292 frames / overlap of 8 frames = 536.5
VIDEO_LENGTH = 536
OVERLAP = 8
CLIP_LENGTH = 16
SCENARIOS = ['Fire Pit', 'Grass Standard', 'Ice Natural', 'Reflection Agent', 'Temple Barbarous']
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

# Returns a random positive image tuple
def get_rnd_pos_img():
    scenario = get_rnd_el(SCENARIOS)
    elevation = get_rnd_el(ELEVATION_ANGLES)
    rotation = get_rnd_el(ROTATION_ANGLES)
    index = randint(1, VIDEO_LENGTH)
    fp_filename = build_filename(True, scenario, index)
    tp_filename = build_filename(False, scenario, index, elevation, rotation)
    return fp_filename, tp_filename

# Returns a random negative inter-video image tuple
def get_rnd_neg_inter_img():
    scenario = get_rnd_el(SCENARIOS)
    index = randint(1, VIDEO_LENGTH)
    fp_filename = build_filename(True, scenario, index)
    scenario = get_rnd_el(SCENARIOS)
    index = randint(1, VIDEO_LENGTH)
    elevation = get_rnd_el(ELEVATION_ANGLES)
    rotation = get_rnd_el(ROTATION_ANGLES)
    tp_filename = build_filename(False, scenario, index, elevation, rotation)
    return fp_filename, tp_filename

# Returns a random negative intra-video image tuple
def get_rnd_neg_intra_img():
    scenario = get_rnd_el(SCENARIOS)
    elevation = get_rnd_el(ELEVATION_ANGLES)
    rotation = get_rnd_el(ROTATION_ANGLES)
    index = randint(1, VIDEO_LENGTH)
    fp_filename = build_filename(True, scenario, index)
    index = randint(1, VIDEO_LENGTH)
    tp_filename = build_filename(False, scenario, index, elevation, rotation)
    fp_image = cv2.imread(fp_filename)
    tp_image = cv2.imread(tp_filename)
    fp_image = cv2.resize(fp_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)
    tp_image = cv2.resize(tp_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)
    dif = ssim(fp_image, tp_image, multichannel=True)
    while dif > INTRA_VIDEO_SSIM_THRESHOLD:
        index = randint(1, VIDEO_LENGTH)
        tp_filename = build_filename(False, scenario, index, elevation, rotation)
        tp_image = cv2.imread(tp_filename)
        tp_image = cv2.resize(tp_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)
        dif = ssim(fp_image, tp_image, multichannel=True)
    return fp_filename, tp_filename

def get_base_filename(filename):
    sub_filename = filename[-10:]
    slash_index = string.find(sub_filename, '/')
    index = int(sub_filename[slash_index + 1:-4]) # all files end in .jpg
    image_file_length = 5 - slash_index + 4
    base_filename = filename[:-image_file_length]
    return base_filename, index


def load_image_tuple(fp_filename, tp_filename):
    # Retrieve the index number
    base_fp_filename, fp_index = get_base_filename(fp_filename)
    base_tp_filename, tp_index = get_base_filename(tp_filename)
    
    # Load the base images from disk
    fp_image = cv2.imread(fp_filename)
    tp_image = cv2.imread(tp_filename)

    # INTER_AREA is the best for image decimation (According to OpenCV documentation)
    fp_image_data = cv2.resize(fp_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)
    tp_image_data = cv2.resize(tp_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)

    # Load an stack the rest on them
    for i in range(1, CLIP_LENGTH):
        #Update the filenames
        fp_filename = base_fp_filename + str(fp_index + i) + '.jpg'
        tp_filename = base_tp_filename + str(tp_index + i) + '.jpg'

        # Load the images from disk
        fp_image = cv2.imread(fp_filename)
        tp_image = cv2.imread(tp_filename)

        # INTER_AREA is the best for image decimation (According to OpenCV documentation)
        fp_image = cv2.resize(fp_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)
        tp_image = cv2.resize(tp_image, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)

        # Stack the images for datum insertion
        fp_image_data = np.dstack((fp_image, fp_image_data))
        tp_image_data = np.dstack((tp_image, tp_image_data))

    image_data = np.dstack((fp_image_data, tp_image_data))
    return image_data

def add_to_db(txn, image_data, label, i):
    # Build the caffe Datum
    datum = caffe.proto.caffe_pb2.Datum()
    datum.height = image_data.shape[0]
    datum.width = image_data.shape[1]
    #Concat the two images using 6 channels
    datum.channels = image_data.shape[2]
    datum.data = image_data.tobytes()
    datum.label = label
    str_id = '{:08}'.format(i)

    # The encode is only essential in Python 3
    txn.put(str_id.encode('ascii'), datum.SerializeToString())

def generate_samples(txn, n_samples, label, get_rnd_img_func):
    for i in range(n_samples):
        # Get a random image
        fp_filename, tp_filename = get_rnd_img_func()
        img_tuple = fp_filename + '-' + tp_filename
        while img_tuple in hit_dict:
            # Get an unused random image
            fp_filename, tp_filename = get_rnd_img_func()
            img_tuple = fp_filename + '-' + tp_filename

        # Add the random image to the used list
        hit_dict[img_tuple] = True

        # Get the images from the disk
        image_data = load_image_tuple(fp_filename, tp_filename)

        # Add to the lmdb
        add_to_db(txn, image_data, label, i)

def create_db(name, n_pos_samples, n_neg_samples):
    env = lmdb.open(name, map_size=MAP_SIZE)
    with env.begin(write=True) as txn:
        # Generate positive samples
        generate_samples(txn, n_pos_samples, POSITIVE_LABEL, get_rnd_pos_img)

        # Generate negative samples
        inter_video_samples = int(n_neg_samples * INTER_VIDEO_RATE)
        intra_video_samples = n_neg_samples - inter_video_samples

        generate_samples(txn, inter_video_samples, NEGATIVE_LABEL, get_rnd_neg_inter_img)
        generate_samples(txn, intra_video_samples, NEGATIVE_LABEL, get_rnd_neg_intra_img)

# Prepare the DBs
hit_dict = {}

#According to the LMDB website: On 64-bit there is no penalty for making this huge (say 1TB)
MAP_SIZE = 1e12

# Create DBs
print 'Generating training database...'
create_db('train', POSITIVE_TRAIN_SAMPLES, NEGATIVE_TRAIN_SAMPLES)
print 'Generating validation database...'
create_db('val', POSITIVE_VAL_SAMPLES, NEGATIVE_VAL_SAMPLES)
print 'Generating testing database...'
create_db('test', POSITIVE_TEST_SAMPLES, NEGATIVE_TEST_SAMPLES)

exit()
