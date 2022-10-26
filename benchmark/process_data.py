import os
import numpy as np
FIELD = None

def format_line(line):
    row = line[:-1].split("\t")
    label = row[0]
    global FIELD
    if FIELD is None:
        FIELD = [str(101 + i) for i, x in enumerate(row[1:])]
    new_line = ' '.join(['{0}:{1}'.format(x, y) for x, y in zip(FIELD, row[1:])])
    new_line = label + "\t" + new_line + "\n"
    return new_line

if __name__ == "__main__":
    from sys import argv
    read_path = argv[1]
    save_train = open(os.path.join(argv[2], "train_sample"), "w")
    save_valid = open(os.path.join(argv[2], "valid_sample"), "w")
    data = []
    for line in open(read_path):
        data.append(format_line(line))

    np.random.shuffle(data)
    n = len(data)
    train_count = int(n * 0.9)
    print (len(data), train_count)
    for line in data[:train_count]:
        save_train.write(line)
    for line in data[train_count:]:
        save_valid.write(line)
    save_train.close()
    save_valid.close()