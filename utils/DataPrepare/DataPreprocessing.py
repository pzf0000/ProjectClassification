from utils.IO.csv import CSV_Input
from utils.DataPrepare.scenario import scenario_choice
from utils.DataPrepare.project_level_name import project_level_name_choice
from utils.DataPrepare.business_group_name import business_group_name_choice
from utils.DataPrepare.project_label import project_label_choice
from utils.DataPrepare.business_unit import business_unit_choice
import numpy as np

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
    pre_data = input.read(filename="SCENARIO.csv")
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
    np.save("data2.npy", data_list)
