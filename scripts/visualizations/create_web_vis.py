import sys
import string
import csv
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


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
    if len(sys.argv) < 4:
        print "Error: Not enough parameters given. Parameters needed: "
        print "-our_neighbors_file -their_neighbors_file -filenames_file -output_html_file"
        exit()
    else:
        OUR_NEIGHBORS_FILE = sys.argv[1]
        THEIR_NEIGHBORS_FILE = sys.argv[2]
        filenames_file = sys.argv[3]
        output_html_file = sys.argv[4]

    our_file = np.load(OUR_NEIGHBORS_FILE)
    our_indices = our_file['arr_0']
    their_file = np.load(THEIR_NEIGHBORS_FILE)
    their_indices = their_file['arr_0']

    N_NEIGHBORS = 5
    DS_ROOT = '/home/lferrer/Documents/Synthetic/'

    # Load the image filenames to print the results
    image_filenames = [line for line in csv.reader(open(filenames_file, "r"))]

    html_code = '<!doctype html><title>Nearest Neighbors Visualization</title><body><table>'
    html_code = html_code + '<tr><td/><td/><td>Our Neighbors</td><td/><td/><td/><td/><td/><td>Their Neighbors</td><td/><td/>'
    same_neighbors = 0
    for i, _ in enumerate(our_indices):
        html_code = html_code + "<tr><td style='width: 200px;'><img src='"
        html_code = html_code + DS_ROOT + image_filenames[i][first_file(i)]
        html_code = html_code + "'/></td>"
        fp_scene, fp_index = parse_label(image_filenames[i][first_label(i)])
        for n in range(N_NEIGHBORS):
            html_code = html_code + "<td style='"
            n_i = our_indices[i][n]
            tp_scene, tp_index = parse_label(image_filenames[n_i][third_label(n_i)])
            if fp_scene == tp_scene:
                if fp_index == tp_index:
                    html_code = html_code + "background-color:#00ff00;"
                else:
                    html_code = html_code + "background-color:#ffff00;"
            for m in range(N_NEIGHBORS):
                if our_indices[i][n] == their_indices[i][m]:
                    html_code = html_code + "border: solid 1px #00f;"
                    same_neighbors += 1
            html_code = html_code + "'><img src='"
            html_code = html_code + DS_ROOT + image_filenames[n_i][third_file(n_i)]
            html_code = html_code + "'/></td>"
        html_code = html_code + "<td style='border-right: solid 1px #f00; border-left: solid 1px #f00;'/>"
        for n in range(N_NEIGHBORS):
            html_code = html_code + "<td style='"
            n_i = their_indices[i][n]
            tp_scene, tp_index = parse_label(image_filenames[n_i][third_label(n_i)])
            if fp_scene == tp_scene:
                if fp_index == tp_index:
                    html_code = html_code + "background-color:#00ff00;"
                else:
                    html_code = html_code + "background-color:#ffff00;"
            for m in range(N_NEIGHBORS):
                if their_indices[i][n] == our_indices[i][m]:
                    html_code = html_code + "border: solid 1px #00f;"
            html_code = html_code + "'><img src='"
            html_code = html_code + DS_ROOT + image_filenames[n_i][third_file(n_i)]
            html_code = html_code + "'/></td>"
        html_code = html_code + "</tr>"
    html_code = html_code + "</table></body>"
    with open(output_html_file, "w") as html_file:
        html_file.write(html_code)
    print same_neighbors, len(our_indices)
