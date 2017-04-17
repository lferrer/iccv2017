import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors

if len(sys.argv) < 3:
    print "Error: Not enough parameters given. Parameters needed: -features_file -filenames_file"
    exit()
else:
    features_file = sys.argv[1]
    filenames_file = sys.argv[2]

# Loading features and computing NearestNeighbors
features = np.load(features_file)
nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(features)
raw_distances, raw_indices = nbrs.kneighbors(features)

# Removing duplicate entries with a 2D array using a NumPy view
aux_distances = np.ascontiguousarray(raw_distances).view(np.dtype((np.void, 
                raw_distances.dtype.itemsize * raw_distances.shape[1])))
distances = np.unique(aux_distances).view(raw_distances.dtype).reshape(-1, raw_distances.shape[1])
aux_indices = np.ascontiguousarray(raw_indices).view(np.dtype((np.void, 
                raw_indices.dtype.itemsize * raw_indices.shape[1])))
indices = np.unique(aux_indices).view(raw_indices.dtype).reshape(-1, raw_indices.shape[1])

min_distances = [1e12 for i in range(10)]
min_indices = [[-1, -1] for i in range(10)]

for index, dist in enumerate(distances):
    for min_dist_index, min_dist in enumerate(min_distances):
        if dist[1] < min_dist:
            min_distances[min_dist_index] = dist[1]
            min_indices[min_dist_index] = indices[index]
            break

# Load the image filenames to print the results
image_filenames = [line.rstrip('\n') for line in open(filenames_file, "r")]

for min_index in min_indices:
    print image_filenames[min_index[0]], image_filenames[min_index[1]]
