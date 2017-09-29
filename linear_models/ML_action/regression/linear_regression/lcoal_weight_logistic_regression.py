import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    num_features = len(open(file_path).readline().split('\t')) - 1
    features = []
    labels = []
    with open(file_path) as data:
        for line in data:
            feature = []
            data_item = line.strip().split('\t')
            for index in range(num_features):
                feature.append(float(data_item[index]))
            features.append(feature)
            labels.append(float(data_item[-1]))
        return features, labels


def sigmoid(eta):
    return 1 / (1 + np.exp(-eta))


def lwlr(features, labels, x, tau, labda, epoch=500):
    X_train = np.mat(features)
    y_train = np.mat(labels).T
    x = np.mat(x)
    m, n = X_train.shape
    theta = np.zeros((n, 1))
    for _ in range(epoch):
        weight = np.exp((- np.sqrt(np.multiply(X_train - x, X_train - x).sum(axis=1))) / (2 * tau ** 2))
        gradient = X_train.T * (np.multiply(weight, y_train - sigmoid(X_train * theta))) - labda * theta
        Hessian = X_train.T * np.mat(np.diag((-1 * np.multiply(weight, np.multiply(sigmoid(X_train * theta), 1 - sigmoid(X_train * theta)))).T.tolist()[0])) * X_train - labda * np.mat(np.eye(2))
        theta = theta - Hessian.I * gradient
    return sigmoid(theta.T * x.T)

print(lwlr([[1,2], [5,4], [4,4], [1,1], [2,2], [2,1], [4,5], [5,5]], [1,0, 0,1, 1,1,0,0], [2,2], 0.1, 0.001))

