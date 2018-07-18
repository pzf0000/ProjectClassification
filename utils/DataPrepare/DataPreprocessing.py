from utils.IO.csv import CSV_Input
from utils.DataPrepare.scenario import scenario_choice
from utils.DataPrepare.project_level_name import project_level_name_choice
from utils.DataPrepare.business_group_name import business_group_name_choice
from utils.DataPrepare.project_label import project_label_choice
import numpy as np

input = CSV_Input()
"""
提取字段：
0   0   * scenario: np.array
3   1   project_id
2   3   project_name
4   5   region_id
5   8   rep_office_id
6   11  customer_id
7   17  project_level_name
8   25  business_group_name
9   26  delivery_type
10  33  project_label
--------------------
1   4   business_unit
"""
if __name__ == '__main__':
    """
    保存数据的格式：
    [[0-2,4-12,[81],]]
    len=[0-12, 13-93(0/1)]
    预测目标索引：2
    """
    pre_data = input.read(filename="db.csv")
    project_id = None
    data_item = [0] * 92
    data_list = []

    for pre_data_item in pre_data:
        new_project_id = pre_data_item[1]

        # 新项目
        if new_project_id != project_id:
            # 上一个项目数据打成numpy
            if project_id is not None:
                data_list.append(np.array(data_item))

            data_item = [0] * 92
            project_id = new_project_id
            # 给各字段赋值
            data_item[0] = pre_data_item[4]
            data_item[1] = pre_data_item[3]  # name
            data_item[2] = int(pre_data_item[1])
            data_item[3] = int(pre_data_item[5])
            data_item[4] = int(pre_data_item[8])
            try:
                data_item[5] = int(pre_data_item[11])
            except:
                data_item[5] = 0
            data_item[7] = project_level_name_choice[pre_data_item[17]]
            data_item[8] = business_group_name_choice[pre_data_item[25]]
            try:
                data_item[9] = int(pre_data_item[26])
            except:
                data_item[9] = 0
            data_item[10] = project_label_choice[pre_data_item[33]]

        scenario = scenario_choice[pre_data_item[0]]
        data_item[10 + scenario] = 1

    # 最后一个项目数据打成numpy
    data_list.append(np.array(data_item))

    # 转换为numpy格式
    data = np.array(data_list)

    # 保存
    np.save("data.npy", data_list)
