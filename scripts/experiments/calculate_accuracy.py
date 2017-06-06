import sys
import numpy as np

def parse_log(log_lines):
    top1_accuracy = 0
    top1_count = 0
    top5_accuracy = 0
    top5_count = 0
    for line in log_lines:
        if line.find('accuracy/top-1 =') > 0:
            sign_index = line.find('=')
            accuracy = float(line[sign_index + 2:])
            top1_accuracy += accuracy
            top1_count += 1
        elif line.find('accuracy/top-5 =') > 0:
            sign_index = line.find('=')
            accuracy = float(line[sign_index + 2:])
            top5_accuracy += accuracy
            top5_count += 1
    top1_accuracy = float(top1_accuracy) / float(top1_count)
    top5_accuracy = float(top5_accuracy) / float(top5_count)
    return top1_accuracy, top5_accuracy

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Error: Not enough parameters given. Parameters needed: "
        print "-log_file"
        exit()
    else:
        LOG_FILE = sys.argv[1]
  
    log_lines = [line.rstrip('\n') for line in open(LOG_FILE, "r")]
    top1_accuracy, top5_accuracy = parse_log(log_lines)
    print 'Top-1 Accuracy: ' + str(top1_accuracy)
    print 'Top-5 Accuracy: ' + str(top5_accuracy)
    
