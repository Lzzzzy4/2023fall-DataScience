# 数据科学导论实验报告

# Predict CO2 Emissions in Rwanda

## 成员

吕思翰 PB21000144

来泽远 PB21000164

曹宸瑞 PB21020659

## 概述

## 数据概览与特性

#### 数据量

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

## 特征分析

## 数据预处理

## 模型训练

## 评估

需要注明的是，由于Kaggle网站提交通道的关闭，我们无法获得标准值。采用的方法是使用Kaggle网站高分开源代码运行获取结果，将其结果作为参考标准。

### 