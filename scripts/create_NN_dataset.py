import string
import os
import sys
import shutil
from random import randint

# Config section of the script
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

def get_random_sample():
    scenario = get_rnd_el(SCENARIOS)
    index = randint(1, VIDEO_LENGTH - CLIP_LENGTH - 1)
    fp_filename = build_filename(True, scenario, index)
    elevation = get_rnd_el(ELEVATION_ANGLES)
    rotation = get_rnd_el(ROTATION_ANGLES)
    tp_filename = build_filename(False, scenario, index, elevation, rotation)
    return fp_filename, tp_filename

def copy_sample(sample_filename):
    slash_index = string.rfind(sample_filename, '/')
    index = int(sample_filename[slash_index + 1:-4])
    sub_dir = sample_filename[len(DS_ROOT):slash_index]
    dest_dir = db_path + sub_dir
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for new_index in range(index, index + CLIP_LENGTH):
        dest_filename = dest_dir + '/' + str(new_index) + ".jpg"
        shutil.copy(sample_filename, dest_filename)

# Create DBs
if len(sys.argv) < 2:
    print "Error: Not enough parameters given. Parameters needed: -db_path -n_samples"
    exit()
else:
    db_path = sys.argv[1]
    n_samples = int(sys.argv[2])

print 'Generating database here: {} with {} samples'.format(db_path, n_samples)

if not os.path.exists(db_path):
    os.makedirs(db_path)

list_file = open(db_path + "/file_list.txt", "w")

for i in range(n_samples):
    fp_filename, tp_filename = get_random_sample()
    while fp_filename in hit_dict:
        fp_filename, tp_filename = get_random_sample()

    # Flag the sample
    hit_dict[fp_filename] = True

    # Copy the files
    copy_sample(fp_filename)
    copy_sample(tp_filename)

    # Add them to the list file
    list_file.write(fp_filename + "\n")
    list_file.write(tp_filename + "\n")
list_file.close()
