## 1.什么是损失函数

用来度量模型的预测值f(x)与真实值Y的差异程度的运算函数

## 2. 为什么使用损失函数

损失函数使用主要是在模型的训练阶段，每个批次的训练数据送入模型后，通过前向传播输出预测值，然后损失函数会计算出预测值和真实值之间的差异值，也就是损失值。得到损失值之后，模型通过反向传播去更新各个参数，来降低真实值与预测值之间的损失，使得模型生成的预测值往真实值方向靠拢，从而达到学习的目的。

## 3. 有哪些损失函数？

### 3.1 基于距离度量的损失函数

基于距离度量的损失函数通常将输入数据映射到基于距离度量的特征空间上，如欧式空间、汉明空间等，将映射后的样本看作空间上的点，采用合适的损失函数度量特征空间上样本真实值和模型预测值之间的距离。特征空间上两个点的距离越小，模型的预测性能越好。

#### 3.1.1 均方误差损失函数(MSE)

公式：L(Y|f(x)) = 1/n Σ(Yi-f(xi))²

在回归问题中，均方误差损失函数用于度量样本点到回归曲线的距离，通过最小化平方损失使样本点可以更好地拟合回归曲线。均方误差损失函数（MSE）的值越小，表示预测模型描述的样本数据具有越好的精确度。由于无参数、计算成本低和具有明确物理意义等优点，MSE已成为一种优秀的距离度量方法。尽管MSE在图像和语音处理方面表现较弱，但它仍是评价信号质量的标准，在回归问题中，MSE常被作为模型的经验损失或算法的性能指标。

代码实现：
     # 自定义实现
      def MSELoss(x:list,y:list):
          """
          x:list，代表模型预测的一组数据
          y:list，代表真实样本对应的一组数据
          """
          assert len(x)==len(y)
          x=np.array(x)
          y=np.array(y)
          loss=np.sum(np.square(x - y)) / len(x)
          return loss
          
#### 3.1.2 L2损失函数(LSE最小平方误差)

公式：L(Y|f(x)) = ꇌ1/n Σ(Yi-f(xi))²

L2损失又被称为欧氏距离，是一种常用的距离度量方法，通常用于度量数据点之间的相似度。由于L2损失具有凸性和可微性，且在独立、同分布的高斯噪声情况下，它能提供最大似然估计，使得它成为回归问题、模式识别、图像处理中最常使用的损失函数。

代码实现：

    # 自定义实现
    def L2Loss(x:list,y:list):
        """
        x:list，代表模型预测的一组数据
        y:list，代表真实样本对应的一组数据
        """
        assert len(x)==len(y)
        x=np.array(x)
        y=np.array(y)
        loss=np.sqrt(np.sum(np.square(x - y)) / len(x))
        return loss
        
#### 3.1.3 L1损失函数(LAD最小绝对值偏差)

公式：L(Y|f(x)) = Σ|Yi-f(xi)|

L1损失又称为曼哈顿距离，表示残差的绝对值之和。L1损失函数对离群点有很好的鲁棒性，但它在残差为零处却不可导。另一个缺点是更新的梯度始终相同，也就是说，即使很小的损失值，梯度也很大，这样不利于模型的收敛。针对它的收敛问题，一般的解决办法是在优化算法中使用变化的学习率，在损失接近最小值时降低学习率

代码实现：

    # 自定义实现
    def L1Loss(x:list,y:list):
        """
        x:list，代表模型预测的一组数据
        y:list，代表真实样本对应的一组数据
        """
        assert len(x)==len(y)
        x=np.array(x)
        y=np.array(y)
        loss=np.sum(np.abs(x - y)) / len(x)
        return loss
       
#### 3.1.4 Smooth L1损失函数

公式：L(Y|f(x)) = (1/2)(Yi-f(xi))² 【|Yi-f(xi)|<1】 ；|Yi-f(xi)|-1/2 【|Yi-f(xi)|>=1】

Smooth L1损失是由Girshick R在Fast R-CNN中提出的，主要用在目标检测中防止梯度爆炸。

代码实现：

    def Smooth_L1(x,y):
        assert len(x)==len(y)
        loss=0
        for i_x,i_y in zip(x,y):
            tmp = abs(i_y-i_x)
            if tmp<1:
                loss+=0.5*(tmp**2)
            else:
                loss+=tmp-0.5
        return loss

#### 3.1.5 huber损失函数

公式：L(Y|f(x)) = (1/2)(Yi-f(xi))² 【|Yi-f(xi)|<= σ】 ；σ|Yi-f(xi)|-(1/2)σ² 【|Yi-f(xi)|>σ】

huber损失是平方损失和绝对损失的综合，它克服了平方损失和绝对损失的缺点，不仅使损失函数具有连续的导数，而且利用MSE梯度随误差减小的特性，可取得更精确的最小值。尽管huber损失对异常点具有更好的鲁棒性，但是，它不仅引入了额外的参数，而且选择合适的参数比较困难，这也增加了训练和调试的工作量。

代码实现：

    delta=1.0  # 先定义超参数

    def huber_loss(x,y):
        assert len(x)==len(y)
        loss=0
        for i_x,i_y in zip(x,y):
            tmp = abs(i_y-i_x)
            if tmp<=delta:
                loss+=0.5*(tmp**2)
            else:
                loss+=tmp*delta-0.5*delta**2
        return loss
