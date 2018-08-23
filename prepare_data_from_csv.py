import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--csv", type=str, default="db.csv")
parser.add_argument("-t", "--npy", type=str, default="data.npy")
args = parser.parse_args()


business_group_name_choice = {
    "Operator": 1,
    "Other": 2,
    "Enterprise": 3,
    "Consumer": 4,
    "1": 5
}

business_unit_choice = {
    None: 0,
    "": 0,
    "Network Integration Services": 1,
    "Software Company Service": 2,
    "Customer Support": 3,
    "Managed Service": 4,
    "IT Integration Service": 5,
    "Network Rollout Service": 6,
    "Managed Service & Network Assurance": 7,
    "SmartCare": 8,
    "Learning Service": 9,
    "Software & Cloud Services": 10,
    "Public Cloud Service": 11,
}

project_label_choice = {
    "": 0,
    "Normal Project": 1,
    "Combined medium/small contracts": 2,
    "Frame plus Volume PO": 3,
}

project_level_name_choice = {
    None: 0,
    "": 0,
    "Top Regional Level (Class A)": 1,
    "Regional Level (Class B)": 2,
    "Top Representative office Level (Class C)": 3,
    "Representative office Level (Class D)": 4,
    "Level D (Rep Office)": 18,
    "Company Level (Class S)": 19,
}

scenario_choice = {
    "408": 1,
    "424": 2,
    "AMI": 3,
    "BCM": 4,
    "Big Data": 5,
    "BigData": 6,
    "Broadcast and Television 700M": 7,
    "BSS": 8,
    "CloudEPN": 9,
    "CloudEPN Business Solution": 10,
    "Competence Consulting and Assessment": 11,
    "Consulting": 12,
    "Core network": 13,
    "CS": 14,
    "DC Facility": 15,
    "DC Infra": 16,
    "DC-L1": 17,
    "DC-L2": 18,
    "DS": 19,
    "EI Delivery": 20,
    "Energy": 21,
    "Energy Pipeline": 22,
    "Enterprise Cloud": 23,
    "Enterprise Cloud Service": 24,
    "Enterprise Network": 25,
    "Enterprise Wireless": 26,
    "Fixed CEM": 27,
    "Fixed network": 28,
    "FTTX/OSP": 29,
    "Government Cloud Service": 30,
    "Home Broadband Business Solution": 31,
    "IBS": 32,
    "IOT": 33,
    "IOT Business Solution": 34,
    "IT Cloud Computing": 35,
    "ITO": 36,
    "KEA (Key Event Assurance)": 37,
    "KEA(Key Event Assurance)": 38,
    "KEA(Key Event ssurance)": 39,
    "Knowledge Transfer": 40,
    "LS": 41,
    "Managed Operation (CT)": 42,
    "Managed Operation (IT)": 43,
    "MBB CEM（Platform+UC）": 44,
    "MBB-RF": 45,
    "MNC": 46,
    "MS": 47,
    "MV-OSS": 48,
    "Network": 49,
    "Network Planning": 50,
    "Network Planning and Optimization Tool": 51,
    "NFV&SDN": 52,
    "NPM/NPI/SQI": 53,
    "NSF(None Sale Fulfillment)": 54,
    "NSQI": 55,
    "OSS": 56,
    "OSS&IES": 57,
    "Planning and Consulting": 58,
    "POD": 59,
    "POD (Pure Equipment)": 60,
    "Public Cloud": 61,
    "Public Cloud Service": 62,
    "Safe City": 63,
    "Smart City": 64,
    "Smart NOS": 65,
    "SOC": 66,
    "Software CS": 67,
    "Software Integration Service": 68,
    "Software ITO MS": 69,
    "Software Revenue Share": 70,
    "SPMS": 71,
    "SQM/CEA": 72,
    "Subscription and Support": 73,
    "Subway eLTE": 74,
    "Telco Cloud": 75,
    "Telco Cloud Business Solution": 76,
    "Transportation": 77,
    "Video": 78,
    "Video Business Solution": 79,
    "Video Platform": 80,
    "Wireless (Including TK)": 81,
}

import numpy as np
import csv


class CSV_Input:
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


input = CSV_Input()

"""
提取字段：
0   3   PROJECT_NAME
1   4   BUSINESS_UNIT
2   5   REGION_ID
3   8   REP_OFFICE_ID
4   11  CUSTOMER_ID
5   17  PROJECT_LEVEL_NAME
6   25  BUSINESS_GROUP_NAME
7   26  DELIVERY_TYPE
8   33  PROJECT_LABEL
"""

if __name__ == '__main__':
    """
    保存数据的格式：
    [[0-8,[81],]]
    len=[0-8, 9-89(0/1)]
    """
    pre_data = input.read(filename=args.csv)
    project_id = None
    data_item = [0] * 90
    data_list = []

    for pre_data_item in pre_data:
        new_project_id = pre_data_item[1]

        # 新项目
        if new_project_id != project_id:
            # 上一个项目数据打成numpy
            if project_id is not None:
                data_list.append(np.array(data_item))

            data_item = [0] * 90
            project_id = new_project_id
            # 给各字段赋值
            data_item[0] = pre_data_item[3]  # PROJECT_NAME
            data_item[1] = business_unit_choice[pre_data_item[6]]  # BUSINESS_UNIT
            data_item[2] = int(pre_data_item[9])  # REGION_ID
            data_item[3] = int(pre_data_item[15])
            try:
                data_item[4] = int(pre_data_item[18])
            except:
                data_item[4] = 0
            data_item[5] = project_level_name_choice[pre_data_item[25]]  # PROJECT_LEVEL_NAME
            data_item[6] = business_group_name_choice[pre_data_item[45]]  # BUSINESS_GROUP_NAME
            try:
                data_item[7] = int(pre_data_item[60])  # DELIVERY_TYPE
            except:
                data_item[7] = 0
            data_item[8] = project_label_choice[pre_data_item[96]]  # PROJECT_LABEL

        scenario = scenario_choice[pre_data_item[0]]
        data_item[8 + scenario] = 1

    # 最后一个项目数据打成numpy
    data_list.append(np.array(data_item))

    # 转换为numpy格式
    data = np.array(data_list)

    # 保存
    np.save(args.npy, data_list)