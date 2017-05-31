import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_log(log_lines, n_iterations, accuracy_rate):
    n_points = n_iterations / accuracy_rate + 1
    top1_accuracy = np.empty((2, n_points))
    top1_accuracy[0] = range(0, n_iterations + accuracy_rate, accuracy_rate)
    top5_accuracy = np.empty((2, n_points))
    top5_accuracy[0] = range(0, n_iterations + accuracy_rate, accuracy_rate)
    for line in log_lines:
        iter_index = line.find('Iteration ')
        if iter_index > 0:
            sub_line = line[iter_index + 10:]
            comma_index = sub_line.find(',')
            iteration = int(sub_line[:comma_index])
        elif line.find('Test net output #0: accuracy/top-1') > 0:
            sign_index = line.find('=')
            accuracy = float(line[sign_index + 2:])
            accuracy_index = iteration / accuracy_rate
            top1_accuracy[1][accuracy_index] = accuracy
        elif line.find('Test net output #1: accuracy/top-5') > 0:
            sign_index = line.find('=')
            accuracy = float(line[sign_index + 2:])
            accuracy_index = iteration / accuracy_rate
            top5_accuracy[1][accuracy_index] = accuracy
    return top1_accuracy, top5_accuracy

def plot_train_test(training_loss, validation_loss, output_file, title):
    label = "Test Label"
    training_color = [0.2, 0.8, 0.1]
    width = 0.75
    training_patch = mpatches.Patch(color=training_color, label='Top-1 Accuracy')
    plt.plot(training_loss[0][5:], training_loss[1][5:], label=label, color=training_color, linewidth=width)
    validation_color = [0.8, 0.2, 0.1]
    plt.plot(validation_loss[0][5:], validation_loss[1][5:], label=label, color=validation_color, linewidth=width)
    validation_patch = mpatches.Patch(color=validation_color, label='Top-5 Accuracy')
    plt.legend(handles=[training_patch, validation_patch])
    plt.title(title)
    plt.savefig(output_file)




if __name__ == '__main__':
    if len(sys.argv) < 6:
        print "Error: Not enough parameters given. Parameters needed: "
        print "-log_file -output_file -n_iterations -accuracy_rate title"
        exit()
    else:
        LOG_FILE = sys.argv[1]
        OUTPUT_FILE = sys.argv[2]
        N_ITERATIONS = int(sys.argv[3])
        ACCURACY_RATE = int(sys.argv[4])
        TITLE = sys.argv[5]

    log_lines = [line.rstrip('\n') for line in open(LOG_FILE, "r")]
    top1_accuracy, top5_accuracy = parse_log(log_lines, N_ITERATIONS, ACCURACY_RATE)
    plot_train_test(top1_accuracy, top5_accuracy, OUTPUT_FILE, TITLE)
    