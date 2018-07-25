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

def data_prepocessing(data_file, project_id=False):
    """
    数据预处理
    :return: 训练集与测试集X / Y
    """
    data = load_dataset(data_file)

    # name = data[:, 0]

    feature_set = []
    label = []

    if not project_id:
        # 全部转为数字
        for data_item in data:
            item = []
            for index in range(len(data_item)):
                if index != 0:
                    item.append(int(data_item[index]))

            feature_set.append([item[0]] + item[2:10])  # 未加入项目名称,去除项目id
            label.append(item[10:])  # 11-91
    else:
        for data_item in data:
            item = []
            feature_set.append(list(data_item[0:2]) + list(data_item[3:11]))  # 去除项目id
            label.append(list(data_item[11:]))  # 11-91
    feature_set = np.array(feature_set)  # 0-8
    label = np.array(label)  # 0-80

    # 将数据随机分成训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(feature_set, label, random_state=1)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':

    data = data_prepocessing("data.npy", False)
    x_train = data[0]
    y_train = data[2]
    x_test = data[1]
    y_test = data[3]

    classifiers = [[GaussianNB(), BinaryRelevance],
                   [BernoulliNB(), LabelPowerset],
                   [DecisionTreeClassifier(criterion="entropy", splitter="random", max_features=0.8, max_depth=18,), LabelPowerset],
                   [SVC(), LabelPowerset],
                   [3, MLkNN]
                   ]
    x = []
    for c in classifiers:
        classifier = c[1](c[0])
        classifier.fit(x_train, y_train)
        start = time.time()
        result = classifier.predict(x_test)
        end = time.time()
        x.append([accuracy_score(y_test, result), end-start])
    print(x)
    """
    data = data_prepocessing("data.npy", True)

    x_train = data[0]
    y_train = data[2]
    x_test = data[1]
    y_test = data[3]

    from sklearn.ensemble import RandomForestClassifier
    from skmultilearn.problem_transform import LabelPowerset
    from skmultilearn.cluster import IGraphLabelCooccurenceClusterer
    from skmultilearn.ensemble import LabelSpacePartitioningClassifier

    # construct base forest classifier
    base_classifier = RandomForestClassifier()

    # setup problem transformation approach with sparse matrices for random forest
    problem_transform_classifier = LabelPowerset(classifier=base_classifier, require_dense=[False, False])

    # partition the label space using fastgreedy community detection
    # on a weighted label co-occurrence graph with self-loops allowed
    clusterer = IGraphLabelCooccurenceClusterer('fastgreedy', weighted=True, include_self_edges=True)

    # setup the ensemble metaclassifier
    classifier = LabelSpacePartitioningClassifier(problem_transform_classifier, clusterer)

    # train
    classifier.fit(x_train, y_train)

    # predict
    predictions = classifier.predict(x_test)
    """