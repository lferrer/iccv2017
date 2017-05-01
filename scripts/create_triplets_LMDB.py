from os import path, makedirs
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
INTER_VIDEO_RATE = 0.80
SAMPLE_PRINT_RATE = 100.0
SAMPLE_BATCH_RATE = 1000.0
CLIP_LENGTH = 16
VIDEO_LENGTH = 4291
NUM_ACTIONS = 62
SCENARIOS = ['Fire Pit', 'Grass Standard', 'Ice Natural', 'Reflection Agent', 'Temple Barbarous',
             'Fire Natural', 'Grass Pit', 'Ice Barbarous', 'Reflection Standard', 'Temple Agent']
ELEVATION_ANGLES = [330, 340, 350]
ROTATION_ANGLES = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, \
                   200, 220, 240, 260, 280, 300, 320, 340]
# Animation classes groups
AIM = [0, 1]
CROUCH_IDLE = [2, 3]
CROUCH_TO_STAND = [4, 5]
CROUCH_WALK = [6, 13]
DEATH = [14, 18]
EQUIP = [19, 20]
FIRE_RIFLE = [21, 22] #Subgroup to represent a short class
FIRE = [21, 24]
HIT_REACT = [25, 28]
IDLE = [29, 33]
JOG = [34, 37]
JUMP = [38, 40]
PRONE = [41, 47]
PRONE_FIRE = [43, 44] #Subgroup to represent a short class
PRONE_TO_STAND = [48, 48]
RELOAD = [49, 53]
SPRINT = [61, 61]
STAND_TO = [54, 56]
WALK = [57, 60]
ALL_ACTIONS = [AIM, CROUCH_IDLE, CROUCH_TO_STAND, CROUCH_WALK, DEATH, EQUIP, FIRE, HIT_REACT,
               IDLE, JOG, JUMP, PRONE, PRONE_TO_STAND, RELOAD, SPRINT, STAND_TO, WALK]
ACTIVE_ACTIONS = [AIM, CROUCH_TO_STAND, CROUCH_WALK, DEATH, JOG, JUMP, PRONE_TO_STAND, SPRINT, STAND_TO, WALK]
PASSIVE_ACTIONS = [CROUCH_IDLE, EQUIP, FIRE, HIT_REACT, IDLE, PRONE, RELOAD]
SHORT_ACTIONS = [FIRE_RIFLE, PRONE_FIRE]
SHORT_ACTIONS_ENDS_INDICES = [21, 22, 43, 44]
ACTION_ENDS = [89, 177, 368, 559, 578, 597, 656, 715, 774, 833, 892, 951, 1010, 1069, 1133, 1195, 1254, 1329,
               1391, 1445, 1498, 1507, 1515, 1555, 1595, 1625, 1655, 1685, 1715, 1906, 2099, 2250, 2376, 2567,
               2613, 2659, 2704, 2750, 2783, 2813, 2852, 2874, 2933, 2941, 2949, 3206, 3307, 3437, 3482, 3534,
               3601, 3668, 3799, 3930, 3955, 3981, 4032, 4089, 4146, 4203, 4260, 4291]
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
def build_filename(first_person, scenario, frame_number, elevation=330, rotation=0):
    image_name = str(frame_number) + '.jpg'
    if first_person:
        filename = path.join('First Person', scenario, image_name)
    else:
        angle_folder = str(elevation) + '-' + str(rotation)
        filename = path.join('Third Person', scenario, angle_folder, image_name)
    return filename

# Returns a random element from a list
def get_rnd_el(my_list):
    return my_list[randint(0, len(my_list) - 1)]

# Returns a random negative sample
def get_rnd_neg_sample(first_person, ref_scenario, same_scenario, ref_action_index):
    if same_scenario:
        scenario = ref_scenario
    else:
        # Get an scenario that is different from the previous one
        scenario = get_rnd_el(SCENARIOS)
        while scenario == ref_scenario:
            scenario = get_rnd_el(SCENARIOS)
    # Get an index that will give us a clip spanning through a single action
    # that is different from the previous one
    frame_number, action_index = get_random_frame()
    if find_in_actions(ref_action_index, ACTIVE_ACTIONS):
        while find_in_actions(action_index, ACTIVE_ACTIONS):
            frame_number, action_index = get_random_frame()
    else:
        while find_in_actions(action_index, PASSIVE_ACTIONS):
            frame_number, action_index = get_random_frame()
    if first_person:
        neg_filename = build_filename(True, scenario, frame_number)
    else:
        elevation = get_rnd_el(ELEVATION_ANGLES)
        rotation = get_rnd_el(ROTATION_ANGLES)
        neg_filename = build_filename(False, scenario, frame_number, elevation, rotation)
    return neg_filename

def get_base_filename(filename):
    sub_filename = filename[-10:]
    slash_index = sub_filename.find('/')
    index = int(sub_filename[slash_index + 1:-4]) # all files end in .jpg
    image_file_length = 5 - slash_index + 4
    base_filename = filename[:-image_file_length]
    return base_filename, index

def get_random_frame():
    # Get a random action that is valid
    action_index = randint(0, NUM_ACTIONS - 1)
    while find_in_actions(action_index, SHORT_ACTIONS):
        action_index = randint(0, NUM_ACTIONS - 1)
    # Get an index that will give us a clip spanning through a single action
    if action_index == 0:
        frame_number = randint(0, ACTION_ENDS[action_index] - CLIP_LENGTH)
    else:
        frame_number = randint(ACTION_ENDS[action_index - 1] + 1, ACTION_ENDS[action_index] - CLIP_LENGTH)
    return frame_number, action_index

def validate_frame(filename):
    # Retrieve the frame number
    base_filename, frame_number = get_base_filename(filename)
    base_filename = path.join(DS_ROOT, base_filename)
    exists = True
    for i in range(CLIP_LENGTH):
        # Update the filenames
        filename = base_filename + str(frame_number + i) + '.jpg'
        # Check for existence
        exists = exists and path.exists( filename)
    return exists

def get_random_triplet(fp_anchor):
    # First get a random scenario
    scenario = get_rnd_el(SCENARIOS)
    # Get a frame_number that will give us a clip spanning through a single action
    frame_number, action_id = get_random_frame()
    # Get the First Person filename
    fp_filename = build_filename(True, scenario, frame_number)
    # Validate the frame
    while not validate_frame(fp_filename):
        frame_number, action_id = get_random_frame()
        fp_filename = build_filename(True, scenario, frame_number)

    # Pick a random Third Person filename
    elevation = get_rnd_el(ELEVATION_ANGLES)
    rotation = get_rnd_el(ROTATION_ANGLES)
    tp_filename = build_filename(False, scenario, frame_number, elevation, rotation)
    # Validate the frame
    while not validate_frame(tp_filename):
        elevation = get_rnd_el(ELEVATION_ANGLES)
        rotation = get_rnd_el(ROTATION_ANGLES)
        tp_filename = build_filename(False, scenario, frame_number, elevation, rotation)

    # Choose if the negative sample is going to be a inter or intra-video sample
    if random() < INTER_VIDEO_RATE:
        negative_filename = get_rnd_neg_sample(not fp_anchor, scenario, False, action_id)
        # Validate the frame
        while not validate_frame(negative_filename):
            negative_filename = get_rnd_neg_sample(not fp_anchor, scenario, False, action_id)
    else:
        negative_filename = get_rnd_neg_sample(not fp_anchor, scenario, True, action_id)
        # Validate the frame
        while not validate_frame(negative_filename):
            negative_filename = get_rnd_neg_sample(not fp_anchor, scenario, True, action_id)

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

def get_action_id(frame_number):
    action_id = 0
    while ACTION_ENDS[action_id] < frame_number:
        action_id = action_id + 1
    return action_id

def find_in_actions(action_id, actions_array):
    for action_limits in actions_array:
        if action_id >= action_limits[0] and action_id <= action_limits[1]:
            return True
    return False

def compute_image_label(image_filename):
    # Remove the person from the filename
    image_filename = image_filename[13:]
    # Get the scene and add it to the label
    slash_index = image_filename.find('/')
    scene = image_filename[:slash_index]
    scene_index = SCENARIOS.index(scene) + 1
    # Get the image number
    image_filename = image_filename[-10:]
    slash_index = image_filename.find('/')
    frame_number = int(image_filename[slash_index + 1:-4]) # all files end in .jpg
    action_id = get_action_id(frame_number)
    return scene_index * 100 + action_id

def load_image_triplet(anchor_filename, positive_filename, negative_filename):
    # Expand the filenames
    anchor_filename = path.join(DS_ROOT, anchor_filename)
    positive_filename = path.join(DS_ROOT, positive_filename)
    negative_filename = path.join(DS_ROOT, negative_filename)

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

def generate_database(db_name, n_samples, index_start):
    # Open log file
    if not path.exists(db_name):
        makedirs(db_name)
    log_file = open(db_name + "/image_list.csv", "w")
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
                img_triplet = "{},{},{}".format(anchor_filename, positive_filename, negative_filename)

                while img_triplet in hit_dict:
                    # Get an unused random triplet
                    anchor_filename, positive_filename, negative_filename = get_random_triplet(fp_anchor)
                    img_triplet = "{},{},{}".format(anchor_filename, positive_filename, negative_filename)

                # Add the random image to the used list
                hit_dict[img_triplet] = True

                # Get the images from the disk
                image_data = load_image_triplet(anchor_filename, positive_filename, negative_filename)

                # Compute the labels
                anchor_label = compute_image_label(anchor_filename)
                negative_label = compute_image_label(negative_filename)
                # Anchor's and Positive's labels are the same
                triplet_label = "{}{}".format(anchor_label, negative_label)

                # Add to the lmdb
                db_index = int(index_list[index + index_start + i])
                add_to_db(txn, image_data, long(triplet_label), db_index)

                # Add the triplet to the log file
                log_file.write(img_triplet + ',')
                log_file.write("{},{}\n".format(anchor_label, negative_label))

                # Invert the anchor to generate another type of sample
                fp_anchor = not fp_anchor

                if (i + 1) % SAMPLE_PRINT_RATE == 0:
                    print "Generated {} out of {} samples".format(index + i + 1, int(n_samples))
        index = index + n_samples_left
    log_file.close()

# Create DBs
if len(sys.argv) < 3:
    print "Error: Not enough parameters given. Parameters needed: -db_path -n_samples"
    exit()
else:
    db_path = sys.argv[1]
    n_samples = int(sys.argv[2])

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
