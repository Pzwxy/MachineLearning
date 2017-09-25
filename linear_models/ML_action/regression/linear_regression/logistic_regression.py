import numpy as np


def load_data(file_path):
    features = []
    labels = []
    with open(file_path) as input_file:
        for line in input_file:
            feature = []
            feature.append(1.0)
            data = line.strip().split('\t')
            for item in data[: -1]:
                feature.append(float(item))
            features.append(feature)
            labels.append(float(data[-1]))
    return features, labels


def sigmoid(eta):
    return 1.0 / (1 + np.exp(-eta))


def gradient_ascent(features, labels, learning_rate=0.001, epoch=500):
    feature_matrix = np.mat(features)
    label_matrix = np.mat(labels).T
    m, n = np.shape(feature_matrix)
    weights = np.ones((n, 1))
    for step in range(epoch):
        h = sigmoid(feature_matrix * weights)
        error = label_matrix - h
        weights = weights + learning_rate * feature_matrix.T * error
    return weights


def stochastic_gradient_ascent(features, labels, learning_rate=0.001):
    feature_matrix = np.mat(features)
    m, n = np.shape(feature_matrix)
    weights = np.ones((n, 1))
    for step in range(m):
        h = sigmoid(features[step] * weights)
        error = labels[step] - h[0, 0]
        weights = weights + learning_rate * error * feature_matrix[step].T
    return weights


def deep_stochastic_gradient_ascent(features, labels, learning_rate=0.001, epoch=500):
    feature_matrix = np.mat(features)
    m, n = np.shape(feature_matrix)
    weights = np.ones((n, 1))
    for batch in range(epoch):
        for step in range(m):
            step = np.random.randint(0, m)
            h = sigmoid(features[step] * weights)
            error = labels[step] - h[0, 0]
            weights = weights + learning_rate * error * feature_matrix[step].T
    return weights


if __name__ == '__main__':
    file_path = 'logistic_regression.txt'
    features, labels = load_data(file_path)
    print(gradient_ascent(features, labels))
    print(stochastic_gradient_ascent(features, labels))
