import numpy as np
import sys

if len(sys.argv) < 3:
    print "Error: Not enough parameters given. Parameters needed: "
    print "-features_file -alpha"
    exit()
else:
    features_file = sys.argv[1]
    ALPHA = float(sys.argv[2])
    N_STREAMS = 3
    
    my_file = np.load(features_file)
    features = my_file['arr_0']
    if N_STREAMS == 3:
        anchor = features[0]
        positive = features[1]
        negative = features[2]
    elif N_STREAMS == 2:
        first_person = features[0]
        third_person = features[1]
    else:
        print "Unsupported number of streams"
        exit()

valid_cases = 0
average_dist = 0
#print np.sum(postive[0]), np.sum(postive[0])
for i, anchor_feature in enumerate(anchor):
    positive_feature = positive[i]
    negative_feature = negative[i]
    dist_same_feature = anchor_feature - positive_feature
    l2_dist_same_feature = np.dot(dist_same_feature, dist_same_feature)
    dist_diff_feature = anchor_feature - negative_feature
    l2_dist_diff_feature = np.dot(dist_diff_feature, dist_diff_feature)
    loss = l2_dist_diff_feature - l2_dist_same_feature
    if loss >= ALPHA:
        valid_cases += 1
    average_dist += loss
print float(valid_cases) / float(len(anchor))
print float(average_dist) / float(len(anchor))
