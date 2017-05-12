import numpy as np
import sys
import csv

def parse_label(label):
    if len(label) > 3:
        scene = label[:2]
    else:
        scene = label[:1]
    action = label[-2:]
    return scene, action

# Returns the column of the filename for third person files
def third_file(index):
    if index % 2 == 0:
        return 1
    else:
        return 0

# Returns the column of the label for third person files
def third_label(index):
    if index % 2 == 0:
        return 4
    else:
        return 3


# Returns the column of the filename for first person files
def first_file(index):
    if index % 2 == 0:
        return 0
    else:
        return 1

# Returns the column of the label for first person files
def first_label(index):
    if index % 2 == 0:
        return 3
    else:
        return 4

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Error: Not enough parameters given. Parameters needed: "
        print "-neighbors_file -filenames_file"
        exit()
    else:
        NEIGHBORS_FILE = sys.argv[1]
        FILENAMES_FILE = sys.argv[2]

    # Neighbors is the first array of a NPZ file
    my_file = np.load(NEIGHBORS_FILE)
    neighbors = my_file['arr_0']

    # Load the image filenames
    image_filenames = [line for line in csv.reader(open(FILENAMES_FILE, "r"))]

    rank_scene_total = 0
    rank_action_total = 0
    rank_frame_total = 0

    for first_index, first_neighbors in enumerate(neighbors):
        rank_scene = 1
        rank_action = 1
        rank_frame = 1
        first_scene, first_action = parse_label(image_filenames[first_index][first_label(first_index)])
        for i, third_index in enumerate(first_neighbors):
            if rank_frame == i + 1:
                if first_index != third_index:
                    rank_frame += 1
            third_scene, third_action = parse_label(image_filenames[third_index][third_label(third_index)])            
            if rank_action == i + 1:
                if first_action != third_action:
                    rank_action += 1
            if rank_scene == i + 1:
                if first_scene != third_scene:
                    rank_scene += 1
        rank_scene_total += rank_scene
        rank_action_total += rank_action
        rank_frame_total += rank_frame

    rank_scene_total = float(rank_scene_total) / float(len(neighbors))
    rank_action_total = float(rank_action_total) / float(len(neighbors))
    rank_frame_total = float(rank_frame_total) / float(len(neighbors))
    print "Scene rank: {}".format(rank_scene_total)
    print "Action rank: {}".format(rank_action_total)
    print "Frame rank: {}".format(rank_frame_total)


