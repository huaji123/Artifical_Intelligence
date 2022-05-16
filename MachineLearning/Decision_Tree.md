# Decision Tree 决策树


from sklearn.tree import DecisionTreeClassifier

1. 定义：是一种描述对实例进行分类的树形结构。由结点（内部节点【一个特征或属性】、叶节点【一个类】）和有向边组成。

2. 流程步骤：特征选择、决策树的生成、决策树的修剪

3. 信息论基础：

      1）信息：消除随机不定性的东西
      
      2）信息熵：随机变量不确定性的度量。H(x) = -Σ P(xi)logP(xi)
      
      3）条件熵：表示在已知随机变量X的条件下随机变量Y的不确定性。H(Y|X) = Σpi H(Y|Xi)
      
      4）信息增益：表示得知特征X的信息而使得Y的信息的不确定性减少的程度
      
      信息增益 = 信息熵-条件熵
      
      集合D的信息熵H(D)与特征A给定条件下D的信息条件熵H(D|A)之差，即公式为： g(D, A) = H(D) - H(D|A)
      
      注：当熵和条件熵的概率由数据估计（特别是极大似然估计）得到时，所对应的熵与条件熵分别称为经验熵(empirical entropy)和经验条件熵(empirical conditional entropy)
