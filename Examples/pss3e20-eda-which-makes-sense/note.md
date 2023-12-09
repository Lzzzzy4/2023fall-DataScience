# note of pss3e20-eda-which-makes-sense

## 经纬度

## 缺失值

少量缺失：用相邻时间点补全

大量缺失：丢掉

## 数据分析

排放：指数分布

按经纬度划分区域后排放：指数分布

某些地区排放多

排放季节周期性

19-20排放量减少，20-21排放量增加

### 特征分析

卫星数据无用，相关性极差（但可以辅助确定趋势）

其他数据（SO2等）理论上无用，空气流动

We'll skip the analysis of these data and note that without the satellite measurements, we have only four features left: latitude, longitude, year and week_no.

### 推测趋势

新冠疫情：2020年的数据应该被视为异常值（或直接忽略）

2022年的CO2排放可以参考其他气体排放

新冠疫情对Q2影响最大

### 维度裁剪

风或大气会导致所有站点数据相似

政治因素会导致所有站点数据相似

自然现象造成季节性

TruncatedSVD降维

四个典型点，The four marked locations are special indeed:

- Nyamasheke has the highest average emission.
- Lake Kivu has high peaks every January and February.
- Kihanga has no seasonal peaks.
- Musanze has seasonal peaks every May and October.

降维后所有点是这四个点（加location）的线性组合

### **Holt-Winters exponential smoothing**

### **Nonnegative matrix factorization**

### 预测

1. 季节性变化
2. 新冠指数衰减
3. 噪音

以上叠加
