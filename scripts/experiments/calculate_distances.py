import sys
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print "Error: Not enough parameters given. Parameters needed: "
        print "-features_file -distances_file -n_streams"
        exit()
    else:
        FEATURES_FILE = sys.argv[1]
        DISTANCES_FILE = sys.argv[2]
        N_STREAMS = int(sys.argv[3])

    my_file = np.load(FEATURES_FILE)
    features = my_file['arr_0']
    if N_STREAMS == 3:
        anchor = features[0]
        positive = features[1]
        fp_anchor = anchor[::2]
        fp_positive = positive[1::2]
        first_person = [val for pair in zip(fp_anchor, fp_positive) for val in pair]
        tp_anchor = anchor[1::2]
        tp_positive = positive[::2]
        third_person = [val for pair in zip(tp_anchor, tp_positive) for val in pair]
    elif N_STREAMS == 2:
        first_person = features[0]
        third_person = features[1]
    else:
        print "Unsupported number of streams"
        exit()

    # Compute the cosine similarity between the features
    distances = pairwise_distances(first_person, Y=third_person, metric='cosine', n_jobs=1)

    # Save the result
    np.savez_compressed(DISTANCES_FILE, distances)
