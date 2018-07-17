import numpy as np
from utils.IO.IO import BaseInput, BaseOutput
import csv


class CSV_Input(BaseInput):
    """
    从csv文件中读取数据
    """

    def input(self, **kwargs):
        """
        重写输入函数
        参数：
        filename: 文件名
        type: 返回数据格式，
            "row": 按行读取
            "col": 按列读取
        retain_title: 是否保留表头，True时不作处理，默认为False，将删除第0行和第0列
        ctn: True/False，索引是否连续，True时index参数只接受长度为2的list，默认为False
        index: 需要返回的数据索引，必须是list，默认为离散的数据，若无type参数则无效
        :param kwargs: 读取参数设置
        :return: numpy.array
        """

        # 必须要有文件路径参数，否则无法读取
        if "filename" in kwargs:
            filename = kwargs["filename"]
        else:
            raise KeyError("no filename.")

        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = np.array(list(reader))
            if "retain_title" in kwargs and kwargs["retain_title"]:
                pass
            else:
                data = np.delete(data, 0, axis=0)
                data = np.delete(data, 0, axis=1)


            # 按照格式返回数据
            if "type" in kwargs:
                read_type = kwargs['type']

                # 按行读取
                if read_type == "row":

                    # 是否需要连续读取，默认离散
                    if "ctn" in kwargs:
                        if kwargs["ctn"]:
                            ctn = True
                        elif not kwargs["ctn"]:
                            ctn = False
                        else:
                            raise KeyError("no such ctn mod.")
                    else:
                        ctn = False

                    # 读取index
                    if "index" in kwargs:
                        index = kwargs["index"]
                    else:
                        raise KeyError("no index.")

                    # 判断index有效
                    if ctn:
                        if isinstance(index, list) and len(index) == 2:
                            a = index[0]
                            b = index[1]
                            if isinstance(a, int) and isinstance(b, int):
                                # 使 a < b
                                if a > b:
                                    a, b = b, a
                            else:
                                raise ValueError("index must be the list of integers.")
                        else:
                            raise ValueError("index is not suitable for continuity.")
                    else:
                        # 离散的情况下，list应该全部为整数，在后续取数过程中再检查
                        pass

                    # 按要求读取
                    if ctn:
                        result = []
                        while a <= b:
                            result.append(data[a])
                            a += 1
                        return np.array(result)
                    else:
                        index.sort()
                        result = []
                        for item in index:
                            if isinstance(item, int):
                                result.append(data[item])
                            else:
                                raise ValueError("index must be the list of integers.")
                        return np.array(result)

                # 按列读取
                elif read_type == "col":
                    # 是否需要连续读取，默认离散
                    if "ctn" in kwargs:
                        if kwargs["ctn"]:
                            ctn = True
                        elif not kwargs["ctn"]:
                            ctn = False
                        else:
                            raise KeyError("no such ctn mod.")
                    else:
                        ctn = False

                    # 读取index
                    if "index" in kwargs:
                        index = kwargs["index"]
                    else:
                        raise KeyError("no index.")

                    # 判断index有效
                    if ctn:
                        if isinstance(index, list) and len(index) == 2:
                            a = index[0]
                            b = index[1]
                            if isinstance(a, int) and isinstance(b, int):
                                # 使 a < b
                                if a > b:
                                    a, b = b, a
                            else:
                                raise ValueError("index must be the list of integers.")
                        else:
                            raise ValueError("index is not suitable for continuity.")
                    else:
                        # 离散的情况下，list应该全部为整数，在后续取数过程中再检查
                        pass

                    # 按要求读取
                    if ctn:
                        result = []
                        while a <= b:
                            result.append(data[:, a])
                            a += 1
                        return np.array(result)
                    else:
                        index.sort()
                        result = []
                        for item in index:
                            if isinstance(item, int):
                                result.append(data[:, item])
                            else:
                                raise ValueError("index must be the list of integers.")
                        return np.array(result)
                else:
                    raise KeyError("no such type.")
            else:
                return data
