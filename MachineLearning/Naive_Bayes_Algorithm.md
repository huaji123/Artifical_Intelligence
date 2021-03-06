# Naive_Bayes_Algorithm

朴素贝叶斯算法：

1.定义：

是基于贝叶斯定理与特征条件独立性假设的分类方法。朴素？假设特征与特征之间相互独立

2.基本公式：
	
	①联合概率分布：表示包含多个条件并且所有的条件都同时成立的概率，记作P(X = a, Y = b) 或 P(a, b) 或 P(ab)

	②P(X∣Y) ：条件概率，又叫似然概率，一般是通过历史数据统计得到。一般不把它叫做先验概率，但从定义上也符合先验定义。

	③相互独立：如果P(a, b) = P(a) * P(b)， 则事件a和事件b相互独立

	④贝叶斯定理：P(Y∣X)= P(X∣Y)P(Y) / P(X)
  
	⑤P(Y)：先验概率。先验概率（prior probability）是指事情还没有发生，求这件事情发生的可能性的大小，是先验概率。它往往作为"由因求果"问题中的"因"出现。
	
	⑥P(Y∣X)：后验概率。后验概率是指事情已经发生，求这件事情发生的原因是由某个因素引起的可能性的大小。后验概率的计算要以先验概率为基础

3. 原理：朴素 + 贝叶斯，

4. 应用场景
	
	文本分类
  
	单词作为特征

5. 拉普拉斯平滑系数： P(F1|C) = (N + a) / (N + am)

	sklearn.naive_bayes.MultinomialNB(alpha = 1.0)

	朴素贝叶斯分类
  
	alpha：拉普拉斯平滑系数

6.总结：

		朴素贝叶斯模型发源于古典数学理论，有稳定的分类效率
    
		对缺失数据不太敏感，算法也比较简单，常用于分本分类
    
		分类准确度高，速度快
    
		由于使用了样本属性独立性的假设，所以如果特征属性有关联时其效果不好
