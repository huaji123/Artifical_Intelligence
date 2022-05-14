# Data_preprocessing:数据预处理

1. 什么是数据预处理？

数据预处理是一种数据挖掘技术，本质就是为了将原始数据转换为可以理解的格式或者符合我们挖掘的格式

2. 为什么需要数据预处理？

真实世界中，数据通常不完整（缺少属性）、不一致（代码或名称差异）、极易受到噪声（错误或异常值）侵扰，低质量的数据将导致低质量的挖掘结果，数据预处理就是解决上面所提到的数据问题的可靠方法

3. 实现

数据清洗=>数据集成=>数据规约=>数据变换
①数据清洗：填写缺失值，光滑噪声数据，识别或删除离群点，并解决不一致性来“清理数据”；
data.dropna(inplace=True)

②数据集成：使用多个数据库，数据立方体或文件；

③数据归约：用替代的、较小的数据表示形式替换元数据，得到信息内容的损失最小值，方法包括维归约、数量归约和数据压缩；

④数据变换：将数据变换成使用挖掘的形式

# 4种预处理常用方法

1. 0均值

2. 归一化

实现：

### 数据归一化处理
### 参数归一化函数
### 归一化时，需要把Dataframe格式转换成ndarray格式；
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        # 根据训练数据集获得数据的均值和方差
        # print(X.shape[1])
        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])
        return self

    def transform(self, X):
        # 将X根据Standardcaler进行均值方差归一化处理
        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / (self.scale_[col])
        return resX

3. 主成分分析

4. 白化
