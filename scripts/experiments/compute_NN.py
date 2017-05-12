import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print "Error: Not enough parameters given. Parameters needed: "
        print "-distances_file -neighbors_file -n_neighbors"
        exit()
    else:
        DISTANCES_FILE = sys.argv[1]
        NEIGHBORS_FILE = sys.argv[2]
        N_NEIGHBORS = int(sys.argv[3])

    # Load the distances
    my_file = np.load(DISTANCES_FILE)
    distances = my_file['arr_0']

    # Store the N_NEIGHBORS smallest distances for each First Person feature
    indices = [[-1 for i in range(N_NEIGHBORS)] for j in range(len(distances))]

    for i, distance_vector in enumerate(distances):
        indices[i] = np.argsort(distance_vector)[:N_NEIGHBORS]
        
     # Save the neighbors
    np.savez_compressed(NEIGHBORS_FILE, indices)
