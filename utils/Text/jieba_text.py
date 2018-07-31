import jieba
from jieba import analyse
from utils.IO.IO import load_dataset
import numpy as np


if __name__ == '__main__':
    data = load_dataset()
    pos_list = []
    neg_list = []

    for index in np.arange(11, 92, 1):
        pos = {}
        neg = {}
        for item in data:
            seg_list = jieba.lcut(item[0])
            if item[index] == "0":
                for seg in seg_list:
                    if seg != " " and seg != "":
                        seg = seg.lower()
                        try:
                            neg[seg] += 1
                        except:
                            neg[seg] = 1
            else:
                for seg in seg_list:
                    if seg != " " and seg != "":
                        seg = seg.lower()
                        try:
                            pos[seg] += 1
                        except:
                            pos[seg] = 1

        # 去重
        pos_new = {}
        for p in pos:
            if p not in neg.keys():
                pos_new[p] = pos[p]

        neg_new = {}
        for n in neg:
            if n not in pos.keys():
                neg_new[n] = neg[n]

        pos = np.array(sorted(pos_new.items(), key=lambda d: d[1], reverse=True))
        neg = np.array(sorted(neg_new.items(), key=lambda d: d[1], reverse=True))

        pos_list.append(pos)
        neg_list.append(neg)
