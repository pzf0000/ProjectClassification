import os
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
import pydotplus


def load_dataset(filename):
    try:
        return np.load(filename)
    except:
        try:
            return pd.read_csv(filename)
        except:
            raise ValueError("Can not open file.")


def train(data, featureset_index=8, label_index=0):
    """
    建立决策树（未加入项目名称）
    :param data: 数据集
    :param featureset_index: 数据集中的feature开始index
    :param label_index: 数据集中label的index
    :return: 决策树
    """
    feature_set = []
    label = []
    for i in data:
        feature_set.append(i[featureset_index:])  # 未加入项目名称
        label.append(i[label_index])

    # 将数据随机分成训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(feature_set, label, random_state=1)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)

    return clf, x_test, y_test


def output_tree_by_pdf(clf):
    """
    将决策树以pdf格式输出
    :param clf: 决策树
    :return: None
    """
    with open("tree.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)

    os.unlink('tree.dot')

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")


def test(clf, x_test, y_test=None):
    """
    预测
    :param clf: 决策树
    :param x_test: 测试集feature
    :param y_test: 测试集label（用来检测准确度，可为空）
    :return: 若提供label，则返回准确度，否则为None
    """
    pre_labels = clf.predict(x_test)

    if y_test is not None:
        x = len(y_test)
        y = 0
        for i in range(x):
            if str(y_test[i]) != str(pre_labels[i]):
                y += 1
        return 1 - y/x


if __name__ == '__main__':
    data = load_dataset("../../../data.npy")
    clf, x_test, y_test = train(data)
    x = test(clf, x_test, y_test)