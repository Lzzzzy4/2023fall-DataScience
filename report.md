# 数据科学导论实验报告

## 比赛名称

- Predict CO2 Emissions in Rwanda

## 成员

吕思翰 PB21000144

来泽远 PB21000164

曹宸瑞 PB21020659

## 问题定义

### Predicting CO2 Emissions

准确监测碳排放能力是应对气候变化的重要步骤。精确的碳排放数据使研究人员和政府能够了解碳排放的来源和模式。尽管欧洲和北美已经建立了广泛的地面碳排放监测系统，但在非洲可用的系统相对较少。本任务要求参赛选手依据过往二氧化碳排放数据预测未来的排放数据。

### Dataset

从卢旺达多个地区挑选了大约497个独特的地点，分布在农田、城市和发电厂周围。这次比赛的数据是按时间划分的；训练数据中包含2019 - 2021年的二氧化碳排放数据，任务是预测2022年至11月的二氧化碳排放数据。

### Evaluation

提交的内容以均方根误差为评分标准。RMSE定义为：$RMSE=\sqrt{\frac{1}{N}\sum\limits_{i=1}^{N}(y_i-\hat y_i)^2}$

## 做题思路

### 数据概览与特性

| #    | Column                                       | Non-Null Count | Dtype   |
| ---- | -------------------------------------------- | -------------- | ------- |
| 0    | ID_LAT_LON_YEAR_WEEK                         | 79023 non-null | object  |
| 1    | latitude                                     | 79023 non-null | float64 |
| 2    | longitude                                    | 79023 non-null | float64 |
| 3    | year                                         | 79023 non-null | int64   |
| 4    | week_no                                      | 79023 non-null | int64   |
| 5    | SulphurDioxide_SO2_column_number_density     | 64414 non-null | float64 |
| 6    | SulphurDioxide_SO2_column_number_density_amf | 64414 non-null | float64 |
| ...  |                                              |                |         |
| 74   | Cloud_solar_zenith_angle                     | 78539 non-null | float64 |
| 75   | emission                                     | 79023 non-null | float64 |

共76行，除经纬度外其余值均为`float64`，不需要做额外数据处理。

部分测试量包含空值，可以进行特殊值处理。

可用作索引的值有`latitude longitude year week_no`，需预测值为`emission`，其余特征

### 特征分析

### 数据预处理

### 模型训练

## 评估

由于课程不在比赛时间内，Kaggle网站提交通道处于关闭状态，我们无法获得标准测试集结果。我们采用的方法是使用Kaggle网站高分开源代码运行获取结果，将其结果作为参考标准，计算RMSE。

## 团队成员分工

- 来泽远
- 吕思翰
- 曹宸瑞

## 个人总结与感悟
