import numpy as np
import matplotlib.pylab as plt


def onehot(categories, category):
    onehot_code = [0] * len(categories)
    onehot_dict = {}
    index = 0
    for cate in categories:
        onehot_dict[cate] = index
        index += 1
    onehot_code[onehot_dict[category]] = 1
    return onehot_code


def softmax_output(result, categories):
    result = result.tolist()
    for category, item in enumerate(result):
        for index, elem in enumerate(item):
            if elem == max(item):
                result[category] = index
                break
    return result


def load_data(file_path, separator='\t'):
    features = []
    labels = []
    with open(file_path) as input_file:
        for line in input_file:
            feature = []
            feature.append(1.0)
            data = line.strip().split(separator)
            for item in data[: -1]:
                feature.append(float(item))
            features.append(feature)
            labels.append(data[-1])
    categories = []
    for label in labels:
        if label not in categories:
            categories.append(label)
    for index, label in enumerate(labels):
        labels[index] = onehot(categories, label)
    return features, labels, categories


def softmax(eta):
    item = np.exp(eta - np.max(eta))
    return item / item.sum(axis=1)


def gradient_ascent(features, labels, categories, learning_rate=0.001, epoch=500):
    feature_matrix = np.mat(features)
    label_matrix = np.mat(labels)
    m, n = np.shape(feature_matrix)
    weights = np.zeros((n, categories))
    for step in range(epoch):
        h = softmax(feature_matrix * weights)
        error = label_matrix - h
        weights = weights + learning_rate * feature_matrix.T * error
    return weights


def stochastic_gradient_ascent(features, labels, categories, learning_rate=0.001):
    feature_matrix = np.mat(features)
    label_matrix = np.mat(labels)
    m, n = np.shape(feature_matrix)
    weights = np.ones((n, categories))
    for step in range(m):
        h = softmax(feature_matrix[step].reshape((1, -1)) * weights)
        error = label_matrix[step].reshape((1, -1)) - h
        weights = weights + learning_rate * feature_matrix[step].reshape((1, -1)).T * error
    return weights


def deep_stochastic_gradient_ascent(features, labels, categories, learning_rate=0.001, epoch=500):
    feature_matrix = np.mat(features)
    label_matrix = np.mat(labels)
    m, n = np.shape(feature_matrix)
    weights = np.ones((n, categories))
    for batch in range(epoch):
        for step in range(m):
            random_m = np.random.randint(0, m)
            h = softmax(feature_matrix[random_m].reshape((1, -1)) * weights)
            error = label_matrix[random_m].reshape((1, -1)) - h
            weights = weights + learning_rate * feature_matrix[random_m].reshape((1, -1)).T * error
    return weights


features, labels, categories = load_data('softmax_regression.txt', separator=',')
# weights = gradient_ascent(features, labels, len(categories))
# print(softmax_output(softmax(np.mat(features * weights)), categories))

weights = deep_stochastic_gradient_ascent(features, labels, len(categories))
print(softmax_output(softmax(np.mat(features * weights)), categories))