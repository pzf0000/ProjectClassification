# 集成交付系统项目分类 / ISDP Project Classification
集成交付系统的项目范围覆盖范围较广，有投标，机会点，运营商交付，企业网等多种类型，又细分网络集成服务，客户支持，培训服务，IT 集成服务等多种业务范畴。  
ISDP 为了灵活支持多种业务定义了多个项目场景，在项目绑定场景后，系统会引导用户使用该场景相关的系统功能，支撑用户完成项目交付。然而项目和场景绑定操作目前由用户手工完成，用户需要从数十个选项的下拉框中选择正确场景，容易引起主观判断或操作失误。  
因此计划对项目基本信息进行学习，实现项目场景分类智能化。  

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

**若需要修改默认的下标值，需要修改原文件 329-347 行**  

### 文件与参数
文件：prepare_data_from_csv.py  
输入：csv文件路径  
输出：npy文件  
参数说明：  

| 参数 | 缩写 | 含义 | 类型 | 默认值 |
| :---: | :---: | --- | :---: | :---: |
| csv | f | csv文件路径（数据库数据需要先保存到csv文件中）| str | "db.csv" |
| npy | t | 目标npy文件路径 | str | "data.npy" |

## 训练、验证与测试 / Train, Verify and Test
文件： train_and_test.py  
输入：数据集  
输出：模型与测试结果  
参数说明：  

| 参数 | 缩写 | 含义 | 类型 | 默认值 |
| :---: | :---: | --- | :---: | :---: |
| batch_size | b | 一个训练批次的大小 | int | 64 |
| dropout | d | 随机节点的比例 | float | 0.5 |
| dataset | data | npy数据的路径 | str | "data.npy" |
| deep_learning | dl | 是否使用深度学习 | bool | True |
| epochs | e | 训练的批次数量 | int | 256 |
| embed_dim | ed | 嵌入操作维度 | int | 128 |
| kernel_sizes | ks | 卷集合大小的集合（字符串需为list形式） | str | "[1, 2, 3, 3, 2, 1]" |
| kernel_num | kn | 卷集合数目 | int | 100 |
| log_interval | l | 每隔多少训练数据打印一次日志 | int | 1 |
| learning_rate | lr | 学习率 | float | 0.002 |
| save_interval | s | 每隔多少固定保存（不包含best的自动存储） | int | 100 |
| save_dir | sd | 模型保存路径（若为None则不保存） | str | "models" |
| static | st | 填充嵌入层 | bool | True |
| save_vocab | sv | 保存词典的路径（包含文件名） | str | None |
| test_interval | t | 每训练多少数据进行一次验证 | int | 100 |
| middle_linear_size | m | 网络中间层节点数量 | int | 7 |
| machine_learning | ml | 是否使用机器学习 | bool | Ture |
| machine_learning_model | mm | 机器学习模型保存路径（为None时不保存模型） | str | "ml_model.npy" |
| class_num | o | 输出的类别数量（标签数量） | int | 81 |

## 预测 / Predict
文件：predict.py  
输入：模型与测试数据  
输出：预测结果  
### 必选参数
- PROJECT_NAME  
- BUSINESS_UNIT  
- REGION_ID
- REP_OFFICE_ID
- CUSTOMER_ID
- PROJECT_LEVEL_NAME
- BUSINESS_GROUP_NAME
- DELIVERY_TYPE
- PROJECT_LABEL
### 可选参数
| 参数 | 缩写 | 含义 | 类型 | 默认值 |
| :---: | :---: | --- | :---: | :---: |
| dropout | d | 随机节点的比例 | float | 0.5 |
| embed_dim | ed | 一个训练批次的大小 | int | 64 |
| embed_dim | ed | 嵌入操作维度 | int | 128 |
| kernel_sizes | ks | 卷集合大小的集合（字符串需为list形式） | str | "[1, 2, 3, 3, 2, 1]" |
| snapshot | s | 深度学习模型路径 | str | "models/best_steps_9500.pt" |
| static | st | 填充嵌入层 | bool | True |
| middle_linear_size | m | 网络中间层节点数量 | int | 7 |
| machine_learning_model | mm | 机器学习模型保存路径（为None时不保存模型） | str | "ml_model.npy" |
| machine_learning_proportion | mp | 机器学习对最终结果的影响比例 | float | 0.5 |
| class_num | o | 输出的类别数量（标签数量） | int | 81 |
| label | l | 是否输出标签，若是则直接输出可能的场景名称，若否则输出01序列 | bool | False |
| vocab | v | 词典路径，需要在深度学习训练过程中保存词典 | str | "text_fields.pt" |
