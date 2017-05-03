import sys
import numpy as np
import string
from sklearn.metrics.pairwise import pairwise_distances

def parse_filename(filename):
    view_less_filename = filename[13:]
    slash_index = view_less_filename.find('/')
    scene = view_less_filename[:slash_index]
    sub_filename = filename[-10:]
    slash_index = string.find(sub_filename, '/')
    index = int(sub_filename[slash_index + 1:-4]) # all files end in .jpg
    return scene, index

def compute_NN(features_file):
    # Loading features and computing NearestNeighbors
    features = np.load(features_file)
    fp_features = features[::2]
    tp_features = features[1::2]

    # Compute the cosine similarity between the features
    distances = pairwise_distances(fp_features, Y=tp_features, metric='cosine', n_jobs=-1)

    # Store the N_NEIGHBORS smallest distances for each First Person feature
    min_distances = [[1e12 for i in range(N_NEIGHBORS)] for j in range(len(fp_features))]
    min_indices = [[-1 for i in range(N_NEIGHBORS)] for j in range(len(fp_features))]

    for index, distance_vector in enumerate(distances):
        for tp_index, dist in enumerate(distance_vector):
            for min_dist_index, min_dist in enumerate(min_distances[index]):
                if dist < min_dist:
                    min_distances[index][min_dist_index] = dist
                    min_indices[index][min_dist_index] = tp_index
                    break

    return min_distances, min_indices


if len(sys.argv) < 5:
    print "Error: Not enough parameters given. Parameters needed: "
    print "-our_features_file -their_features_file -filenames_file -output_html_file"
    exit()
else:
    our_features_file = sys.argv[1]
    their_features_file = sys.argv[2]
    filenames_file = sys.argv[3]
    output_html_file = sys.argv[4]

N_NEIGHBORS = 5
DS_ROOT = '/home/lferrer/Documents/Synthetic/NN/'

# Compute the N_NEIGHBORS for both features
our_distances, our_indices = compute_NN(our_features_file)
their_distances, their_indices = compute_NN(their_features_file)

# Load the image filenames to print the results
image_filenames = [line.rstrip('\n') for line in open(filenames_file, "r")]

html_code = '<!doctype html><title>Nearest Neighbors Visualization</title><body><table>'
html_code = html_code + '<tr><td/><td/><td>Our Neighbors</td><td/><td/><td/><td/><td/><td>Their Neighbors</td><td/><td/>'
for index in range(len(our_indices)):
    html_code = html_code + "<tr><td style='width: 200px;'><img src='"
    html_code = html_code + DS_ROOT + image_filenames[2*index]
    html_code = html_code + "'/></td>"
    fp_scene, fp_index = parse_filename(image_filenames[2*index])
    for n in range(N_NEIGHBORS):
        html_code = html_code + "<td style='"
        tp_scene, tp_index = parse_filename(image_filenames[2*our_indices[index][n] + 1])
        if fp_scene == tp_scene:
            if abs(fp_index - tp_index) <= 16:
                html_code = html_code + "background-color:#00ff00;"
            else:
                html_code = html_code + "background-color:#ffff00;"
        for m in range(N_NEIGHBORS):
            if our_indices[index][n] == their_indices[index][m]:
                html_code = html_code + "border: solid 1px #00f;"
        html_code = html_code + "'><img src='"
        html_code = html_code + DS_ROOT + image_filenames[2*our_indices[index][n] + 1]
        html_code = html_code + "'/></td>"
    html_code = html_code + "<td style='border-right: solid 1px #f00; border-left: solid 1px #f00;'/>"
    for n in range(N_NEIGHBORS):
        html_code = html_code + "<td style='"
        tp_scene, tp_index = parse_filename(image_filenames[2*their_indices[index][n] + 1])
        if fp_scene == tp_scene:
            if abs(fp_index - tp_index) <= 16:
                html_code = html_code + "background-color:#00ff00;"
            else:
                html_code = html_code + "background-color:#ffff00;"
        for m in range(N_NEIGHBORS):
            if their_indices[index][n] == our_indices[index][m]:
                html_code = html_code + "border: solid 1px #00f;"
        html_code = html_code + "'><img src='"
        html_code = html_code + DS_ROOT + image_filenames[2*their_indices[index][n] + 1]
        html_code = html_code + "'/></td>"
    html_code = html_code + "</tr>"
html_code = html_code + "</table></body>"
with open(output_html_file, "w") as html_file:
    html_file.write(html_code)
