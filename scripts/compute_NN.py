import sys
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

if len(sys.argv) < 3:
    print "Error: Not enough parameters given. Parameters needed: -features_file -filenames_file"
    exit()
else:
    features_file = sys.argv[1]
    filenames_file = sys.argv[2]

N_DISTANCES = 5
N_SAMPLES = 10

# Loading features and computing NearestNeighbors
features = np.load(features_file) 
fp_features = features[::2]
tp_features = features[1::2]

# Compute the cosine similarity between the features
distances = pairwise_distances(fp_features, Y=tp_features, metric='cosine', n_jobs=-1)

# Store the N_DISTANCES smallest distances for each First Person feature
min_distances_full = [[1e12 for i in range(N_DISTANCES)] for j in range(len(fp_features))]
min_indices_full = [[-1 for i in range(N_DISTANCES)] for j in range(len(fp_features))]

for index, distance_vector in enumerate(distances):
    for tp_index, dist in enumerate(distance_vector):
        for min_dist_index, min_dist in enumerate(min_distances_full[index]):
            if dist < min_dist:
                min_distances_full[index][min_dist_index] = dist
                min_indices_full[index][min_dist_index] = tp_index
                break

# Get the best N_SAMPLES from the First Person sample frames
min_distances = [1e12 for i in range(N_SAMPLES)]
min_indices = [-1 for i in range(N_SAMPLES)]

for index, distance_vector in enumerate(min_distances_full):
    dist_sum = np.sum(distance_vector)
    for min_dist_index, min_dist in enumerate(min_distances):
        if dist_sum < min_dist:
            min_distances[min_dist_index] = dist_sum
            min_indices[min_dist_index] = index
            break

# Load the image filenames to print the results
image_filenames = [line.rstrip('\n') for line in open(filenames_file, "r")]

for min_index in min_indices:
    print image_filenames[min_index*2]
    for min_index_tp in min_indices_full[min_index]:
        print image_filenames[min_index_tp*2 + 1]
