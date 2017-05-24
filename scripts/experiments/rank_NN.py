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

def get_frame_number(filename):
    sub_filename = filename[-10:]
    slash_index = sub_filename.find('/')
    index = int(sub_filename[slash_index + 1:-4]) # all files end in .jpg
    return index

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
        found_scene = False
        found_frame = False
        found_action = False
        list_scene = []
        list_frame = []
        list_action = []
        first_scene, first_action = parse_label(image_filenames[first_index][first_label(first_index)])
        first_frame = get_frame_number(image_filenames[first_index][first_file(first_index)])
        for i, third_index in enumerate(first_neighbors):
            if not found_frame:
                third_frame = get_frame_number(image_filenames[third_index][third_file(third_index)])
                if third_frame not in list_frame:
                    list_frame.append(third_frame)
                if abs(third_frame - first_frame) <= 8:
                    found_frame = True                    
            third_scene, third_action = parse_label(image_filenames[third_index][third_label(third_index)])            
            if not found_action:
                if third_action not in list_action:
                    list_action.append(third_action)
                if first_action == third_action:
                    found_action = True
            if not found_scene:
                if third_scene not in list_scene:
                    list_scene.append(third_scene)
                if first_scene == third_scene:
                    found_scene = True
        rank_scene_total += len(list_scene)
        rank_action_total += len(list_action)
        rank_frame_total += len(list_frame)

    rank_scene_total = float(rank_scene_total) / float(len(neighbors))
    rank_action_total = float(rank_action_total) / float(len(neighbors))
    rank_frame_total = float(rank_frame_total) / float(len(neighbors))
    print "Scene rank: {}".format(rank_scene_total)
    print "Action rank: {}".format(rank_action_total)
    print "Frame rank: {}".format(rank_frame_total)


