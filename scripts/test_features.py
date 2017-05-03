import numpy as np
import sys

if len(sys.argv) < 2:
    print "Error: Not enough parameters given. Parameters needed: "
    print "-features_file"
    exit()
else:
    features_file = sys.argv[1]
    
# Loading features 
features = np.load(features_file)
fp_features = features[::2]
tp_features = features[1::2]

valid_cases = 0
for i, fp_feature in enumerate(fp_features):
    same_tp_feature = tp_features[i]
    dist_same_feature = np.linalg.norm(fp_feature - same_tp_feature)**2
    valid = 0
    for j, tp_feature in enumerate(tp_features):
        if i != j:
            dist = np.linalg.norm(fp_feature - tp_feature)**2
            if dist > dist_same_feature:
                valid = valid + 1
    valid_cases = valid_cases + valid
print float(valid_cases) / float(len(fp_features)**2)
