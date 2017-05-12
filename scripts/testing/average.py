import numpy as np
import sys

if len(sys.argv) < 2:
    print "Error: Not enough parameters given. Parameters needed: "
    print "-features_file"
    exit()
else:
    FEATURES_FILE = sys.argv[1]
    N_STREAMS = 3
    
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

total = 0
for i, fp in enumerate(first_person):
    tp = third_person[i]
    total += np.linalg.norm(fp - tp)**2
print float(total) / float(len(first_person))
