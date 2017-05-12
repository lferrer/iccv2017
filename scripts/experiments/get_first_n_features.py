import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print "Error: Not enough parameters given. Parameters needed: "
        print "-input_file -output_file -n"
        exit()
    else:
        INPUT_FILE = sys.argv[1]
        OUTPUT_FILE = sys.argv[2]
        N = int(sys.argv[3])

    my_file = np.load(INPUT_FILE)
    features = my_file['arr_0']
    
    subset = [i[:N] for i in features]
    subset = np.asarray(subset)
    np.savez_compressed(OUTPUT_FILE, subset)
