from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import pandas as pd


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

        feature_set.append(np.array([item[0]] + item[2:9]))  # 未加入项目名称
        label.append(np.array(item[90]))  # 10-90

    feature_set = np.array(feature_set)
    label = np.array(label)

    # 将数据随机分成训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(feature_set, label, random_state=1)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    clf = SVC()
    clf.fit(x_train, y_train)
    result = clf.predict(x_test)

    w = 0
    for i in y_test:
        if i == 0:
            w += 1

    y = len(y_test)
    x = 0
    for i in range(y):
        # if abs(y_test[i] - result[i]) < 0.1:
        #     x += 1
        if y_test[i] == result[i]:
            x += 1

    print(w / y)
    print(x / y)
    print(np.mean(result == y_test))
    a = np.array([result, y_test])
