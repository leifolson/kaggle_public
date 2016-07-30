"""
Module for loading mnist data and converting to lists
"""

def load_train_data():
    """
    Loads the mnist training data and returns data with labels as lists

    :return:
    train_data, train_label
    """
    with open('/Users/clint/Development/data/mnist/train.csv', 'r') as reader:
        reader.readline()
        train_label = []
        train_data = []
        for line in reader.readlines():
            data = list(map(int, line.rstrip().split(',')))
            train_label.append(data[0])
            train_data.append(data[1:])

    return train_data, train_label


def load_test_data():
    with open('/Users/clint/Development/data/mnist/test.csv', 'r') as reader:
        reader.readline()
        test_data = []
        for line in reader.readlines():
            pixels = list(map(int, line.rstrip().split(',')))
            test_data.append(pixels)

    return test_data