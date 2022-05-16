# RandomForest 随机森林

1. 定义： 通过集成学习的思想将多颗决策树集成的一种算法，是一个分类器

  tips:决策树的数量越大，随机森林算法的鲁棒性越强，精确度越高

2. 生成原理（两次随机选取）：

        1）样本随机：假设训练数据集共有M个对象的数据，从样本数据中采取有放回（Bootstrap）随机抽取N个样本（因为是有放回抽取，有些数据可能被选中多次，有些数据可能不被选中），每次取出的样本不完全相同，这些样本组成了决策树的训练数据集
        
        2）特征随机：假设每个样本数据都有 K 个特征，从所有特征中随机地选取 k 个特征，选择最佳分割属性作为节点建立CART决策树，决策树成长期间 k 的大小始终不变（在Python中构造随机森林模型的时候，默认取特征的个数 k 是 K 的平方根）；

3. 优缺点：

        1）优点：
               
               （1）模型准确率高，可以处理分类和回归问题，即使存在部分数据缺失，也能保持高精度
               
               （2）能够处理数量庞大的高纬度的特征，且不需要进行降维（因为特征子集是随机选择的）
               
               （3）能够评估各个特征在分类问题上的重要性：可以生成树状结构，判断各个特征的重要性
               
               （4）对异常值、缺失值不敏感
               
               （5）随机森林有袋外数据（OOB），不需要交叉验证
        
        2）缺点：
            
                （1）解决回归问题的效果不如分类问题
                
                （2）树之间的相关性越大，错误率就越大
                 
                （3）当训练数据噪声较大时，容易产生过拟合现象
           
 4. 应用场景

        （1）银行利用随机森林来寻找忠诚度高和忠诚度低的客户
        
        （2）医药行业用随机森林来寻找正确的成分组合以获得新药，随机森林也可以对病人的记录进行分析从而确诊病情
        
        （3）随机森林也应用在电子商务的推荐引擎中以确定客户对推荐的好评度

 5. 相关知识

         （1）集成学习：组合这里的多个弱监督模型以期得到一个更好更全面的强监督模型，集成学习潜在的思想是即便某一个弱分类器得到了错误的预测，其他的弱分类器也可以将错误纠正回来。

         （2）Bagging(bootstrap aggregating)：
         
                    bootstrap：自助法，一种由放回的抽样方法；目的：得到统计量的分布以及置信区间
                    
                    在Bagging方法中，利用bootstrap方法从整体数据集中采取有放回抽样得到N个数据集，在每个数据集上学习出一个模型，最后的预测结果利用N个模型的输出得到，具体地：分类问题采用N个模型预测投票的方式，回归问题采用N个模型预测平均的方式。
                    
         （3）Boosting（提升方法）：一种用来减小监督学习中偏差的机器学习算法。代表：AdaBoost（Adaptive boosting）
          
         （4）Stacking方法：训练一个模型用于组合其他各个模型，通常使用logistic回归作为组合策略
 
 6. 相关问题



 7. 代码实现：

     class RFClassifier:
        '''
        随机森林回归器
        '''

        def __init__(self, n_estimators=100, random_state=0):
            # 随机森林的大小
            self.n_estimators = n_estimators
            # 随机森林的随机种子
            self.random_state = random_state

        def fit(self, X, y):
            '''
            随机森林分类器拟合
            :param X:
            :param y:
            :return:
            '''
            # 获取y的种类
            self.y_classes = np.unique(y)
            # 决策树数组
            dts = []
            n = X.shape[0]
            rs = np.random.RandomState(self.random_state)
            for i in range(self.n_estimators):
                # 创建决策树分类器
                dt = DecisionTreeClassifier(random_state=rs.randint(np.iinfo(np.int32).max), max_features="auto")
                # 根据随机生成的权重，拟合数据集
                dt.fit(X, y, sample_weight=np.bincount(rs.randint(0, n, n), minlength=n))
                dts.append(dt)

            self.trees = dts

        def predict(self, X):
            '''
            随机森林分类器预测
            :param X:
            :return:
            '''

            # 预测结果数组
            probas = np.zeros((X.shape[0], len(self.y_classes)))

            for i in range(self.n_estimators):
                # 决策树分类器
                dt = self.trees[i]
                # 依次预测结果可能性
                probas += dt.predict_proba(X)
            # 预测结果可能性平均
            probas /= self.n_estimators
            # 返回预测结果
            return self.y_classes.take(np.argmax(probas, axis=1), axis=0)

    import numpy as np
    from sklearn.tree import DecisionTreeRegressor

    class RFRegressor:
        """
        随机森林回归器
        """

        def __init__(self, n_estimators = 100, random_state = 0):
            # 随机森林的大小
            self.n_estimators = n_estimators
            # 随机森林的随机种子
            self.random_state = random_state

        def fit(self, X, y):
            """
            随机森林回归器拟合
            """
            # 决策树数组
            dts = []
            n = X.shape[0]
            rs = np.random.RandomState(self.random_state)
            for i in range(self.n_estimators):
                # 创建决策树回归器
                dt = DecisionTreeRegressor(random_state=rs.randint(np.iinfo(np.int32).max), max_features = "auto")
                # 根据随机生成的权重，拟合数据集
                dt.fit(X, y, sample_weight=np.bincount(rs.randint(0, n, n), minlength = n))
                dts.append(dt)
            self.trees = dts

        def predict(self, X):
            """
            随机森林回归器预测
            """
            # 预测结果
            ys = np.zeros(X.shape[0])
            for i in range(self.n_estimators):
                # 决策树回归器
                dt = self.trees[i]
                # 依次预测结果
                ys += dt.predict(X)
            # 预测结果取平均
            ys /= self.n_estimators
            return ys
