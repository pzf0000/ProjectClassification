from utils.IO.csv import CSV_Input
from utils.DataPrepare.scenario import scenario_choice
import numpy as np

input = CSV_Input()
"""
提取字段：
0   0   * scenario: np.array
1   1   project_id
2   3   project_name
4   5   region_id
5   8   rep_office_id
6   11  customer_id
7   14  contry_code
8   16  representive_office_name
9   17  project_level_name
10  25  business_group_name
11  26  delivery_type
12  29  creation_time
13  33  project_label
--------------------
3   4   business_unit
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
    data_item = [0] * 94
    data_list = []

    for pre_data_item in pre_data:
        new_project_id = pre_data_item[1]

        # 新项目
        if new_project_id != project_id:
            # 上一个项目数据打成numpy
            if project_id is not None:
                data_list.append(np.array(data_item))

            data_item = [0] * 94
            project_id = new_project_id
            # 给各字段赋值
            data_item[0] = pre_data_item[1]
            data_item[1] = pre_data_item[3]
            data_item[2] = pre_data_item[4]
            data_item[3] = pre_data_item[5]
            data_item[4] = pre_data_item[8]
            data_item[5] = pre_data_item[11]
            data_item[6] = pre_data_item[14]
            data_item[7] = pre_data_item[16]
            data_item[8] = pre_data_item[17]
            data_item[9] = pre_data_item[25]
            data_item[10] = pre_data_item[26]
            data_item[11] = pre_data_item[29]
            data_item[12] = pre_data_item[33]

        scenario = scenario_choice[pre_data_item[0]]
        data_item[12 + scenario] = 1

    # 最后一个项目数据打成numpy
    data_list.append(np.array(data_item))

    # 转换为numpy格式
    data = np.array(data_list)

    # 保存
    np.save("data.npy", data_list)
