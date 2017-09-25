import numpy as np
from matplotlib import pyplot as plt


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


def regression_with_normal_equation(features, labels):
    x = np.mat(features)
    y = np.mat(labels).T
    x_T_x = x.T * x
    if np.linalg.det(x_T_x) == 0:
        print("Regression with normal equation failure!")
        return
    w = x_T_x.I * x.T * y
    return w

if __name__ == '__main__':
    file_path = 'linear_regression.txt'
    features, labels = load_data(file_path)
    w = regression_with_normal_equation(features, labels)
    y_ = np.mat(features) * w
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.mat(features)[:, 1], np.mat(labels).T)
    ax.plot(np.mat(features)[:, 1], y_)
    plt.show()
    print("The corrcoef of the model is:")
    print(np.corrcoef(y_.T, np.mat(labels)))