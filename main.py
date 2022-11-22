import math
import operator
from collections import Counter
import bitstring
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from numpy import linalg as LA


def euclidean(idx: int, X_test, X_train, y_train, k: int, **kwargs):
    voting = kwargs.get('voting', False)
    weights = kwargs.get('weights', None)
    if weights:
        np.multiply(X_test[idx], weights, out=X_test[idx])
    distances = []
    for i in range(len(X_train)):
        d = 0
        for j in range(len(X_train[i])):
            d += (X_train[i][j] - X_test[idx][j]) ** 2
        distances.append(math.sqrt(d))

    sort_index = np.argsort(distances)
    if not voting:
        return sort_index[1:k + 1]
    else:
        weights = dict.fromkeys(set(y_train), 0)
        for i in sort_index:
            if distances[i] != 0:
                weights[y_train[i]] += 1 / distances[i] ** 2
        return weights


def manhattan(idx: int, X_test, X_train, y_train, k: int, **kwargs):
    voting = kwargs.get('voting', False)
    weights = kwargs.get('weights', None)
    if weights:
        np.multiply(X_test[idx], weights, out=X_test[idx])
    distances = []
    for i in range(len(X_train)):
        d = 0
        for i_, j_ in zip(X_train[i], X_test[idx]):
            d += abs(i_ - j_)
        distances.append(d)
    sort_index = np.argsort(distances)
    if not voting:
        return sort_index[1:k + 1]
    else:
        weights = dict.fromkeys(set(y_train), 0)
        for i in sort_index:
            if distances[i] != 0:
                weights[y_train[i]] += 1 / distances[i] ** 2
        return weights


def cosine(idx: int, X_test, X_train, y_train, k: int, **kwargs):
    voting = kwargs.get('voting', False)
    weights = kwargs.get('weights', None)
    if weights:
        np.multiply(X_test[idx], weights, out=X_test[idx])
    distances = []
    for i in range(len(X_train)):
        d = 1 - (LA.norm(X_train[i]) * LA.norm(X_test[idx])) / np.dot(X_train[i],
                                                                      X_test[idx])
        distances.append(d)
    sort_index = np.argsort(distances)
    if not voting:
        return sort_index[1:k + 1]
    else:
        weights = dict.fromkeys(set(y_train), 0)
        for i in sort_index:
            if distances[i] != 0:
                weights[y_train[i]] += 1 / distances[i] ** 2
        return weights


def chebyshev(idx: int, X_test, X_train, y_train, k: int, **kwargs):
    voting = kwargs.get('voting', False)
    weights = kwargs.get('weights', None)
    if weights:
        np.multiply(X_test[idx], weights, out=X_test[idx])
    distances = []
    for i in range(len(X_train)):
        d = []
        for i_, j_ in zip(X_train[i], X_test[idx]):
            d.append(abs(i_ - j_))
        distances.append(max(d))
    sort_index = np.argsort(distances)
    if not voting:
        return sort_index[1:k + 1]
    else:
        weights = dict.fromkeys(set(y_train), 0)
        for i in sort_index:
            if distances[i] != 0:
                weights[y_train[i]] += 1 / distances[i] ** 2
        return weights


def hamming(idx: int, X_test, X_train, y_train, k: int, **kwargs):
    voting = kwargs.get('voting', False)
    weights = kwargs.get('weights', None)
    if weights:
        np.multiply(X_test[idx], weights, out=X_test[idx])
    distances = []
    np.multiply(X_test[idx], weights, out=X_test[idx])

    for i in range(len(X_train)):
        d = 0
        for i_, j_ in zip(X_train[i], X_test[idx]):
            d += sum(i != j for i, j in
                     zip(bitstring.BitArray(float=i_, length=32), bitstring.BitArray(float=j_, length=32)))
        distances.append(d)
    sort_index = np.argsort(distances)
    if not voting:
        return sort_index[1:k + 1]
    else:
        weights = dict.fromkeys(set(y_train), 0)
        for i in sort_index:
            if distances[i] != 0:
                weights[y_train[i]] += 1 / distances[i] ** 2
        return weights


def accuracy(y_true, y_pred):
    scores = [int(y_true[i] == y_pred[i]) for i in range(len(y_true))]
    return scores.count(1) / len(scores)


def standardize(df, cols_to_scale):
    for i in cols_to_scale:
        df[i] = (df[i] - df[i].mean()) / df[i].std()
    return df


def find_optimal_k(X_train, y_train, X_test, y_test):
    """K lays within the range [1, 120]"""
    error_rate = []
    for i in range(1, 6):
        cls = KNeighborsClassifier(n_neighbors=i)
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        error_rate.append(1 - accuracy_score(y_test, y_pred))
    # print(f"Min error rate: {min(error_rate)} k: {error_rate.index(min(error_rate))+1}")
    return error_rate.index(min(error_rate)) + 1
    # plt.plot(range(1, len(X_train) + 1), error_rate)
    # plt.show()


class KNN_classifier:
    def __init__(self, metric=euclidean, k=5):
        self.metric = metric
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test, **kwargs):
        voting = kwargs.get('voting', False)
        weights = kwargs.get('weights', None)
        y_preds = []
        for i in range(len(X_test)):
            nearest = self.metric(i, X_test, self.X_train, self.y_train, self.k, voting=voting, weights=weights)
            if voting:
                y_preds.append(max(nearest.items(), key=operator.itemgetter(1))[0])
            else:
                y_pred = max(self.y_train[nearest])
                y_preds.append(y_pred)
        return y_preds


if __name__ == '__main__':
    df = pd.read_csv('../lab10.1NN/Iris.csv')
    cols_to_scale = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    df = standardize(df, cols_to_scale)

    """Нахождение оптимального k перебором"""
    optimal_k = []
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(df[cols_to_scale].values, df['Species'].values,
                                                            test_size=0.2)
        optimal_k.append(find_optimal_k(X_train, y_train, X_test, y_test))

    print(Counter(optimal_k).most_common())


    # X_train, X_test, y_train, y_test = train_test_split(df[cols_to_scale].values, df['Species'].values,
    #                                                     test_size=0.2)
    # cls = KNN_classifier(k=7)
    # cls.fit(X_train, y_train)
    # y_pred = cls.predict(X_test, voting=True)
    # print(accuracy(y_test, y_pred))
    #
    # knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    # knn.fit(X_train, y_train)
    # y_pred = knn.predict(X_test)
    # print(accuracy_score(y_test, y_pred))
    #
    # df_validate = pd.read_csv('Dop (1).csv')
    # cols_to_scale_ = ['Sepal.L', 'Sepal.W', 'Petal.L', 'Petal.W']
    #
    # features_ = StandardScaler().fit_transform(df_validate[cols_to_scale_].values)
    # df_validate[cols_to_scale_] = features_
    # y_preds = cls.predict(df_validate.values)
    # print(y_preds)
