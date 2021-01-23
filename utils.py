import numpy as np


def label_encoder(label):
    """
    encode output which is a number in range [0,i] into a vector of size i+1
    :param label: output vector to be encoded
    :return: encoded matrix represent the value of each vector value in terms of zeroes and ones
    """
    labels = set([label[i][0] for i in range(len(label))])
    encoded_label = np.zeros([len(label), len(labels)])
    for i in range(len(label)):
        index = list(labels).index(label[i][0])
        encoded_label[i][index] = 1

    return encoded_label


def batch_normalization(data, epsilon=1e-6):
    """
    normalize data with mean and variance
    :param data: data to be normalized
    :param epsilon: epsilon value, default value is 10^-6
    :return: normalized data
    """
    normalized_data = []
    for i in range(len(data)):
        normalized_sample = []
        for j in range(len(data[i])):
            x = data[i][j] - data[i].mean(axis=0) / np.sqrt(data[i].var(axis=0)+epsilon)
            normalized_sample.append(x)
        normalized_data.append(list(normalized_sample))
    return np.array(normalized_data)


def zero_pad(x, pad_width, dims):
    """
    for convolution padding
    Pads the given array x with zeroes at the both end of given dims.
    :param x: array to be padded
    :param pad_width: width of the padding
    :param dims: dimensions to be padded
    :return: x_padded -> zero padded x
    """
    dims = dims if isinstance(dims, int) else dims
    pad = [(0, 0) if idx not in dims else (pad_width, pad_width) for idx in range(len(x.shape))]
    x_padded = np.pad(x, pad, 'constant')
    return x_padded
