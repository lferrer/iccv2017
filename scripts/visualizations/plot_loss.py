import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_log(log_lines, n_iterations, display_rate, test_rate):
    n_training_points = n_iterations / display_rate
    training_loss = np.empty((2, n_training_points))
    training_loss[0] = range(0, n_iterations, display_rate)
    n_validation_points = n_iterations / test_rate + 1
    validation_loss = np.empty((2, n_validation_points))
    validation_loss[0] = range(0, n_iterations + test_rate, test_rate)
    for line in log_lines:
        iter_index = line.find('Iteration ')
        if iter_index > 0:
            sub_line = line[iter_index + 10:]
            comma_index = sub_line.find(',')
            iteration = int(sub_line[:comma_index])
        elif line.find('Test net output #0') > 0:
            loss_index = line.find('loss = ')
            sub_line = line[loss_index + 7:]
            space_index = sub_line.find(' ')
            loss = float(sub_line[:space_index])
            validation_index = iteration / test_rate
            validation_loss[1][validation_index] = loss
        elif line.find('Train net output #0') > 0:
            loss_index = line.find('loss = ')
            sub_line = line[loss_index + 7:]
            space_index = sub_line.find(' ')
            loss = float(sub_line[:space_index])
            training_index = iteration / display_rate
            training_loss[1][training_index] = loss
    return training_loss, validation_loss

def plot_train_test(training_loss, validation_loss, output_file, title):
    label = "Test Label"
    training_color = [0.2, 0.8, 0.1]
    width = 0.75
    training_patch = mpatches.Patch(color=training_color, label='Training Loss')
    plt.plot(training_loss[0][5:], training_loss[1][5:], label=label, color=training_color, linewidth=width)
    validation_color = [0.8, 0.2, 0.1]
    plt.plot(validation_loss[0][5:], validation_loss[1][5:], label=label, color=validation_color, linewidth=width)
    validation_patch = mpatches.Patch(color=validation_color, label='Validation Loss')
    plt.legend(handles=[training_patch, validation_patch])
    plt.title(title)
    plt.savefig(output_file)




if __name__ == '__main__':
    if len(sys.argv) < 7:
        print "Error: Not enough parameters given. Parameters needed: "
        print "-log_file -output_file -n_iterations -display_rate -test_rate -title"
        exit()
    else:
        LOG_FILE = sys.argv[1]
        OUTPUT_FILE = sys.argv[2]
        N_ITERATIONS = int(sys.argv[3])
        DISPLAY_RATE = int(sys.argv[4])
        TEST_RATE = int(sys.argv[5])
        TITLE = sys.argv[6]

    log_lines = [line.rstrip('\n') for line in open(LOG_FILE, "r")]
    training_loss, validation_loss = parse_log(log_lines, N_ITERATIONS, DISPLAY_RATE, TEST_RATE)
    plot_train_test(training_loss, validation_loss, OUTPUT_FILE, TITLE)
    