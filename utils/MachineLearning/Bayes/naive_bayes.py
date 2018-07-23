import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

def load_dataset(filename):
    try:
        return np.load(filename)
    except:
        try:
            return pd.read_csv(filename)
        except:
            raise ValueError("Can not open file.")


if __name__ == '__main__':
    data = load_dataset("data.npy")
    feature_set = []
    label = []

    for data_item in data:
        # 转为数字
        item = []
        for index in range(len(data_item)):
            if index != 0:
                item.append(int(data_item[index]))

        feature_set.append(np.array(item[0:9]))  # 未加入项目名称
        label.append(np.array(item[80]))  # 11-91

    feature_set = np.array(feature_set)
    label = np.array(label)

    # 将数据随机分成训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(feature_set, label, random_state=1)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    clf = GaussianNB()
    # clf = BernoulliNB()
    clf.fit(x_train, y_train)

    result = clf.predict(x_test)

    w = 0
    for i in y_test:
        if i == 1:
            w += 1
    x = 0
    y = len(result)
    result *= 10

    for i in range(y):
        # if abs(y_test[i] - 10*result[i]) < 0.1:
        #     x += 1
        if y_test[i] == result[i]:
            x += 1

    print(1 - w / y)
    print(x / y)
    print(np.mean(result == y_test))

    # 准确率与召回率
    precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
    result = clf.predict_proba(x_test)[:, 1]
    report = result > 0.5
    print(classification_report(y_test, report, target_names=['thin', 'fat']))
    a = np.array([result, y_test])

    plt.plot(result, y_test, '*')
    plt.show()
