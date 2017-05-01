from os import path
from random import randint, random

CLIP_LENGTH = 16
VIDEO_LENGTH = 4291
INTER_VIDEO_RATE = 0.80
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

def find_in_actions(action_id, actions_array):
    for action_limits in actions_array:
        if action_id >= action_limits[0] and action_id <= action_limits[1]:
            return True
    return False

def get_base_filename(filename):
    sub_filename = filename[-10:]
    slash_index = sub_filename.find('/')
    index = int(sub_filename[slash_index + 1:-4]) # all files end in .jpg
    image_file_length = 5 - slash_index + 4
    base_filename = filename[:-image_file_length]
    return base_filename, index

DS_ROOT = '/home/lferrer/Documents/Synthetic'

def validate_frame(filename):
    # Retrieve the frame number
    base_filename, frame_number = get_base_filename(filename)
    base_filename = path.join(DS_ROOT, base_filename)
    exists = True
    for i in range(CLIP_LENGTH):
        # Update the filenames
        filename = base_filename + str(frame_number + i) + '.jpg'
        # Check for existence
        exists = exists and path.exists(filename)
    return exists

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

t = get_random_triplet(True)
t2 = get_random_triplet(False)