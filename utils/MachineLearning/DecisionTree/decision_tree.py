import os

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
import pydotplus

from utils.IO.IO import load_dataset

def train_feature(
        data,
        label_index,
        criterion="entropy",
        splitter="random",
        max_features=0.8,
        max_depth=18,
        min_samples_split=2,
        min_samples_leaf=1
):
    """
    建立决策树（未加入项目名称）
    :param min_samples_leaf:
    :param min_samples_split:
    :param max_depth: 决策树最大深度
    :param max_features: 最大的特征数
    :param criterion: 建树方式“gini”/“entropy”
    :param splitter: “best”/“random”
    :param data: 数据集
    :return: 决策树
    """
    feature_set = []
    label = []

    for data_item in data:
        # 转为数字
        item = []
        for index in range(len(data_item)):
            if index != 0:
                item.append(int(data_item[index]))

        feature_set.append(item[0:10])  # 未加入项目名称
        label.append(item[11:])  # 11-91

    feature_set = np.array(feature_set)
    label = np.array(label)

    # 将数据随机分成训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(feature_set, label, random_state=1)

    clf = tree.DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    clf = clf.fit(x_train, y_train)

    return clf, x_test, y_test


def output_tree_by_pdf(clf, filename="tree"):
    """
    将决策树以pdf格式输出
    :param filename: 输出文件名称
    :param clf: 决策树
    :return: None
    """
    with open(filename + ".dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)

    os.unlink(filename + ".dot")

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(filename + ".pdf")


def test_feature(clf, x_test, proba=False, y_test=None):
    """
    预测
    :param clf: 决策树
    :param proba: 是否需要输出概率
    :param x_test: 测试集feature
    :param y_test: 测试集label（用来检测准确度，可为空）
    :return: 若提供label，则返回准确度，否则返回预测结果
    """
    if proba:
        pre_labels = clf.predict_proba(x_test)
    else:
        pre_labels = clf.predict(x_test)

    if y_test is not None and not proba:
        x = len(y_test)
        y = 0
        for i in range(x):
            if str(y_test[i]) != str(pre_labels[i]):
                y += 1
        return 1 - y / x

    return pre_labels


def train(data, save_model_name=None):
    """
    训练数据
    :param data: 数据集
    :param save_model_name: 保存模型文件名称，若空则不保存模型
    :return: 决策树数组
    """
    clf_list = []
    for i in np.arange(11, 92, 1):
        clf, x_test, y_test = train_feature(data, i)
        clf_list.append(clf)
    if save_model_name is not None:
        joblib.dump(clf_list, save_model_name)
    return clf_list


def load_model(file_name):
    return joblib.load(file_name)


def test(feature_set, label, clf_list, proba=False):
    """
    测试数据
    :param feature_set: 测试数据X
    :param label: 测试数据Y
    :param clf_list: 决策树数组（从文件导入后）
    :param proba: 是否输出概率，默认为False
    :return:
    """
    result = []
    try:
        for i in feature_set:
            item = []
            for index in np.arange(11, 92, 1):
                x = test_feature(clf_list[index - 11], feature_set, y_test=label, proba=proba)
                item.append(x)
            result.append(item)
    except:
        # 若仅有一项
        item = []
        for index in np.arange(11, 92, 1):
            x = test_feature(clf_list[index - 11], [feature_set], y_test=[label], proba=proba)
            item.append(x)
        result.append(item)
    return np.array(result)


def predict(data, clf_list, proba=False):
    """
    预测，不需要输入测试数据的标签
    :param data: 数据集
    :param clf_list: 决策树模型
    :param proba:
    :return:
    """
    result = []
    try:
        for X in data:
            item = []
            for index in np.arange(11, 92, 1):
                x = test_feature(clf_list[index - 11], [X], proba=proba)
                item.append(x)
            result.append(item)
        return np.array(result)
    except:
        # 仅有一项
        item = []
        for index in np.arange(11, 92, 1):
            x = test_feature(clf_list[index - 11], [data], proba=proba)
            item.append(x)
        result.append(item)
        return np.array(result)


if __name__ == '__main__':
    data = load_dataset("../../../data.npy")

    feature_set = []
    label = []

    for data_item in data:
        # 转为数字
        item = []
        for index in range(len(data_item)):
            if index != 0:
                item.append(int(data_item[index]))

        feature_set.append(item[0:10])  # 未加入项目名称
        label.append(item[11:])  # 11-91

    feature_set = np.array(feature_set)
    label = np.array(label)

    train(data, "decision_tree.m")

    clf_list = load_model("decision_tree.m")

    result = test(feature_set, label, clf_list=clf_list, proba=False)
