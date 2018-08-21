# 项目分类 / Project Classification
## 数据预处理 / Derpare Data
### 数据字段说明

|下标|字段名|
|:---:|:---:|
|0|PROJECT_NAME|
|1|BUSINESS_UNIT|
|2|REGION_ID|
|3|REP_OFFICE_ID|
|4|CUSTOMER_ID|
|5|PROJECT_LEVEL_NAME|
|6|BUSINESS_GROUP_NAME|
|7|DELIVERY_TYPE|
|8|PROJECT_LABEL|
|9-89|场景 0 / 1 序列

### 文件与参数
文件：prepare_data_from_csv.py  
参数说明：  

| 参数 | 缩写 | 含义 | 类型 | 默认值 |
| :---: | :---: | --- | :---: | :---: |
| csv | f | csv文件路径（数据库数据需要先保存到csv文件中）| str | "db.csv" |
| npy | t | 目标npy文件路径 | str | "data.npy" |

## 训练、验证与测试 / Train, Verify and Test
文件： train_and_test.py  
参数说明：  

| 参数 | 缩写 | 含义 | 类型 | 默认值 |
| :---: | :---: | --- | :---: | :---: |
| batch_size | b | 一个训练批次的大小 | int | 64 |
| dropout | d | 随机节点的比例 | float | 0.2 |
| dataset | data | npy数据的路径 | str | "data.npy" |
| epochs | e | 训练的批次数量 | int | 256 |
| embed_dim | ed | 嵌入操作维度 | int | 128 |
| kernel_sizes | ks | 卷集合大小的集合（字符串需为list形式） | str | "[1, 2, 3, 2, 1]" |
| kernel_num | kn | 卷集合数目 | int | 100 |
| log_interval | l | 每隔多少训练数据打印一次日志 | int | 1 |
| learning_rate | lr | 学习率 | float | 0.001 |
| save_interval | s | 每隔多少固定保存（不包含best的自动存储） | int | 100 |
| save_dir | sd | 模型保存路径（若为None则不保存） | str | "model" |
| static | st | 填充嵌入层 | bool | True |
| test_interval | t | 每训练多少数据进行一次验证 | int | 100 |
| middle_linear_size | m | 网络中间层节点数量 | int | 8 |
| class_num | o | 输出的类别数量（标签数量） | int | 81 |

