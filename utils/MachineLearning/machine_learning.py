import numpy as np
import time
from sklearn.model_selection import train_test_split
from utils.IO.IO import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def data_prepocessing(data_file):
    """
    数据预处理
    :return: 训练集与测试集X / Y
    """
    data = load_dataset(data_file)

    feature_set = []
    label = []

    # 全部转为数字
    for data_item in data:
        item = []
        for index in range(len(data_item)):
            if index != 0:
                item.append(int(data_item[index]))

        feature_set.append(item[:8])  # 未加入项目名称,去除项目id
        label.append(item[8:])  # 9-89

    feature_set = np.array(feature_set)
    label = np.array(label)

    # 将数据随机分成训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(feature_set, label, random_state=1)

    return x_train, x_test, y_train, y_test


def train(data, save=False, name="ml_model.npy"):
    x_train = data[0]
    y_train = data[2]

    classifiers = [
        # [BernoulliNB(), LabelPowerset],
        [DecisionTreeClassifier(criterion="entropy", splitter="random", max_features=0.8, max_depth=18), LabelPowerset],
        [DecisionTreeClassifier(criterion="entropy", splitter="random", max_features=0.8, max_depth=16), LabelPowerset],
        [DecisionTreeClassifier(criterion="gini", splitter="best", max_features=0.8, max_depth=12), LabelPowerset],
        [DecisionTreeClassifier(criterion="gini", splitter="best", max_features=0.8, max_depth=14), LabelPowerset],
        [DecisionTreeClassifier(), LabelPowerset],
        # [SVC(), LabelPowerset],
        # [3, MLkNN],
    ]

    for i in range(len(classifiers)):
        classifiers[i] = classifiers[i][1](classifiers[i][0])
        classifiers[i].fit(x_train, y_train)

    classifiers = np.array(classifiers)

    if save:
        np.save(name, classifiers)

    return classifiers


def load_model(path="ml_model.npy"):
    return np.load(path)


def test(classifiers, x_test, y_test=None, sum=True, weight=None):
    if x_test.ndim == 1:
        x_test = x_test.reshape(1, -1)
        try:
            y_test = y_test.reshape(1, -1)
        except:
            pass

    acc = []
    results = []
    for c in classifiers:
        result = c.predict(x_test)
        result = result.toarray()
        if y_test is not None:
            delta = result - y_test
            size = len(delta)
            corrects = 0.0
            for i in range(size):
                s = 0
                correct = 0
                for j in range(len(delta[i])):
                    if result[i][j] == 1 or y_test[i][j] == 1:
                        s += 1
                        if int(result[i][j]) == int(data[3][i][j]):
                            correct += 1
                corrects += float(correct) / float(s)

            corrects /= size
            # acc.append(accuracy_score(y_test, result))
            acc.append(corrects)
        results.append(result)

    if sum:
        result = [[0] * 81] * len(results[0])
        if weight is None:
            weight = [1] * len(results[0])
        for i in range(len(results)):
            result += results[i] * weight[i]
        results = result

    if y_test is None:
        return results
    else:
        return acc, results


if __name__ == '__main__':
    data = data_prepocessing("../../data2.npy")
    train(data, True)
    classifiers = load_model()

    weight = np.array([1]*len(classifiers))
    start = time.time()
    result = test(classifiers, data[1][0], data[3][0], weight=weight)
    end = time.time()
    print(end-start)
    acc_list = result[0]
    result = result[1]

    result = result / weight.sum()
    size = len(result[0])
    corrects = 0.0
    for i in range(size):
        sum = 0
        correct = 0
        for j in range(len(result[i])):
            if result[i][j] == 1 or data[3][i][j] == 1:
                sum += 1
                if round(result[i][j]) == round(data[3][i][j]):
                    correct += 1
        corrects += float(correct) / float(sum)
    corrects /= size
    print(acc_list)
    print(corrects)
