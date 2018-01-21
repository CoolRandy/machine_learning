## Machine Learning

[TOC]

### 机器学习科普

参考文章： http://nbviewer.jupyter.org/github/zlotus/notes-LSJU-machine-learning/blob/master/ReadMe.ipynb

http://mp.weixin.qq.com/s/Ad22EUAu8VAhy5AVlqLBDw

机器学习定义：

> A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E
>
> 一个程序被认为能从经验 E 中学习，解决任务 T，达到性能度量值
> P，当且仅当，有了经验 E 后，经过 P 评判，程序在处理 T 时的性能有所提升

机器学习主要解决的是两类问题：监督学习和非监督学习

如何掌握解决这两类问题的基本思路？

关键是掌握“套路”：
1、如何把现实场景中的问题抽象成相应的数学模型，并知道在这个抽象过程中，数学模型有怎样的假设。

2、如何利用数学工具，对相应的数学模型参数进行求解。

3、如何根据实际问题提出评估方案，对应用的数学模型进行评估，看是否解决了实际问题。

自学路径：

1、实践完整的机器学习流程：包括数据收集、清洗、预处理，建立模型，调整参数和模型评估

2、在真实的数据集中练习，逐渐建立哪种模型适合哪种挑战的直觉

3、深入到一个具体主题中，例如在数据集中应用不同类型的聚类算法，看哪些效果最好

机器学习的主要任务就是分类。采用某个机器学习算法分类：1、做算法训练，即学习如何分类。通常我们为算法输入大量已分类数据作为算法的训练集。训练集是用于训练机器学习算法的数据样本集合。每个训练样本包含特征（feature）、目标变量等。目标变量是机器学习算法的预测结果，在分类算法中目标变量的类型通常是标称型的，而在回归算法中通常是连续型的。

> 特征或者属性通常是训练样本集的列，他们是独立测量得到的结果，多个特征联系在一起共同组成一个训练样本。

为测试机器学习算法效果，通常使用两套独立的样本集：训练数据和测试数据。

机器学习还有一个任务是回归，主要用于预测数值型数据。回归的经典样例就是数据拟合曲线：通过给定数据点的最优拟合曲线。

上述的分类和回归都属于监督学习（这类算法必须知道预测什么，即目标变量的分类信息）

相对应的无监督学习，此时数据没有类别信息，也不会给定目标值。在无监督学习中，将数据集合分成由类似的对象组成的多个类的过程称之为聚类；将寻找描述数据统计值的过程称之为密度估计。

现在很多问题都可以采用多种算法来解决，那针对同一个问题该如何选择合适的算法呢？

**选择实际可用算法必须考虑两个问题：**

1、使用机器学习算法的目的，想要算法完成何种任务；2、需要分析和搜集的数据是什么

对于问题1，如果想要预测目标变量的值，则可以选择监督学习算法，否则可以选择无监督学习算法。确定监督学习算法之后，需要进一步确定目标变量类型，如果目标变量是离散型的，可以选择分类算法；如果目标变量是连续型的，则需要选择回归。

同样的对于无监督学习算法，同样需要进一步分析是否需要将数据划分为离散的组。如果这是唯一的要求，则使用聚类算法；如果还需要估计数据和每个分组的相似程度，则还需要使用密度估计算法。

接下来就是要考虑问题2了，需要充分了解数据，主要了解数据的一下特性：

特征值是离散型还是连续型变量，特征值中是否存在缺失的值，何种原因造成的缺失，数据中是否存在异常值（这就是数据清洗的操作了），某个特征发生的频率等。

【机器学习的个人理解】

**以电商为例，简单来说就是用户浏览页面商品的点击数据（历史浏览数据），结合某些特征维度进行训练，这个训练的过程是由具体的算法来执行，最终会生成一个预测模型，这个预测模型就是从这些历史数据中挖掘出来一些规律，之后对于新的用户以及新的商品，就能够根据某些特征来进行用户喜好，可能性购买的一种预测，进而作出个性化的推荐。**

这里机器学习的模型到底是个什么东西？人们往往在学习某个东西之后会产生自己的技能，但是这个技能是个抽象的东西，对应到机器学习的模型，就很难想象它到底可以具象成一个什么东西；实际上，简单理解，这个学习得到的模型就是**映射**。从数学的角度来理解就是函数，给定输入集合的一个元素，函数会唯一的对应一个输出值。

比如函数f(x)=w_1*x_1+w_2*x_2+...+w_d*x_d+b，当w_1,w_2,...,w_d是确定的，那么给定一组x_1,x_2,...,x_d，就能唯一确定一个输出值f(x)。

那对于一个电商页面而言，比如手淘的有好货，机器学习学习的模型就是这样一个函数：

> 给定一个用户和商品，这个函数就能够唯一输出一个分数，表示用户点击该商品的可能性。

知道了机器学习的概念，那接下里就要考虑机器学习的应用场景了。

什么样的问题需要用到机器学习来解决，而不是直接采用常用的算法呢？简单回答就是：**难以用规则解决的问题可以尝试用机器学习来解决**

算法导论中经典的排序问题，无论是快排还是归并排序，解法都是一些确定的规则；而对于机器学习问题，比如垃圾邮件识别、识别一张图片上的物体是不是树叶等，这些就很难用一个确定的规则来解决，前者是由于很难穷举，而后者则是由于很难去描述树叶的规则。

由前面的历史浏览数据来训练模型的实例可以看出**历史数据中隐藏着用户是否会点击某个商品的某种规律**。所以机器学习应用的一个必要条件就是：**有大量数据，并且数据中有隐藏的某种规律或模式**

对于没有任何规律的事情，再多的数据也是没用的

总结下来会发现，机器学习的三要素就是：数据、学习算法和模型（映射）

![样本数据](/Users/randy/Documents/ml/machine_learning/resource/样本数据.jpg)

接下来针对具体的示例来说明：

上图展示的是一份样本数据，该训练数据的每一行称为一个训练样本；注意到每个样本包含三个属性：年龄、性别和商品价格。代表了我们认为用户是否会点击该商品取决于这三个因素，但实际上影响用户点击的因素远不止这三个因素，这里是简化处理。这些属性我们统一称之为**特征（feature）**。在该场景中，我们还需要对用户是否点击商品进行预判，这个预判结果也需要记入这个模型系统之中，因此是否点击这个信息被记做**标注（label）**。

上述这份训练数据是从哪里来的？绝大多数互联网产品都会把用户的行为数据——包括浏览历史、点击历史记录下来，我们称为**日志（Log）**。从日志数据中就能知道每个用户点过什么商品（对应标注为1的样本），看了什么商品却没有点（对应标注为-1的样本），再关联上用户的特征数据（年龄、性别）和商品的特征数据（价格），就得到学习算法所需要的训练数据了

上面提及的标注有无可以将机器学习问题大致分为监督和无监督两类。

- 监督学习：每个输入样本都有标注，这些标注就像老师的标准答案一样”监督“着学习的过程。而监督学习又大致分成两类：**分类（Classification）**和**回归（Regression）**：

- - 分类问题：标注是离散值，比如用户”点击“和”不点击“。如果标注只有两个值，则称为二分类，如果标注有多个值，则称为多分类。
  - 回归问题：标注是连续值，比如如果问题是预测北京市房屋的价格，价格作为标注就是一个连续值，属于回归问题。

- 无监督学习：训练样本没有标注，无监督学习解决的典型问题是**聚类（clustering）**问题。比如对一个网站的用户进行聚类，看看这个网站用户的大致构成，分析下每类用户群的特点是什么。

![映射](/Users/randy/Documents/ml/machine_learning/resource/映射.png)

数据部分的映射：

首先，我们可以假设一个完美的映射f，它不仅能够对训练数据中的所有样本都能够正确的预测用户是否点击，对于遇到的新的样本也是一样的。但是现实中并不存在这样的完美模型（**Ground Truth**），也称之为目标模型，**也即这个完美模型（函数）是未知的**。既然不存在，那我们如何以它为目标来学习呢？这时我们就只能依赖训练数据，当训练数据足够多的时候，我们就可以认为海量的样本反映了Ground Truth f的样子，我们就称这种情况为训练数据来自于f。

![机器学习细化](/Users/randy/Documents/ml/machine_learning/resource/机器学习细化.jpg)

假设了Ground Truth f的存在，那么学习算法要做的就是找出某个映射，使得这个映射尽可能的接近f。在实际训练过程中，学习算法会有一个**假设集合(Hypothesis Set，记作H)**，这个集合包含所有候选的映射函数。学习算法做的事情就是从中选出最好的g，使得g越接近f越好。

这里可以用一个示例来说明这个假设函数的情况：

![假设函数示例](/Users/randy/Documents/ml/machine_learning/resource/假设函数示例.png)

银行核定是否发放信用卡给客户，那可能的公式（假设函数）是上图中的三个：1、年收入是否超过80万台币； 2、负债超过10万台币； 3、工作不满两年

这个假设函数集合可能包含好的或不好的，最终要筛选出来一个最好的作为g

> Machine Learning：Use training data to compute mode g  that approximates Ground Truth f.

接下来我们来学习一个学习算法：**PLA**，全称 Perceptron Learning Algorithm。其中 Perceptron 译作**感知机**，它是人工神经网络中最基础的两层神经网络模型。

经典的PLA算法问题存在着两点性质：

1、算法可能永远也无法运行结束，会迷失在茫茫的训练数据中永远找不到出口

2、哪怕知道PLA最终能找到出口，我们也无法事先知道学习需要花多久

但是对于上述两个问题可以通过算法的升级版加以解决

PLA的基本思路：

**每个特征都有一个权重Wi表示该特征的重要程度，综合所有的特征和权重计算一个最终的分数，如果分数超过某个阈值（threshold），就表示用户会点击，否则不会点击。**

![用户日志训练数据](/Users/randy/Documents/ml/machine_learning/resource/用户日志训练数据.jpg)

如上图所示，每一条样本表示为x=(x1, x2),其中x1表示年龄，x2表示商品价格。样本标注用y表示，y=1表示用户点击，y=-1表示用户没有点击。

我们将特征的权重记作w = (w1, w2)，w1代表了年龄这维特征的重要性，w2代表商品价格这维特征的重要性。于是判断一个用户会不会点击，就变成了下面这个函数：
$$
y = \left\{\begin{matrix}
 1,& if \sum_{i=1}^{n}w_{i}x_{i}>threshold \\ 
 -1,& if \sum_{i=1}^{n}w_{i}x_{i}<threshold& 
\end{matrix}\right.
$$
将该函数进行简单的变换，可以采用符号函数sign(x)来表示：

当x大于0， sign(x)等于1；当x小于0， sign(x)等于-1
$$
h(x) = sign(\sum_{i=1}^{n}({\color{Red} w_{i}}x_{i})-{\color{Red} threshold})
$$
注：在算法完成学习之前，函数中的Wi和threshold是未知的，不同的![w_i](http://www.zhihu.com/equation?tex=w_i)和![threshold](http://www.zhihu.com/equation?tex=threshold)值对应了不同的函数，事实上**所有可能的Wi和threshold所代表的函数集合，构成了PLA的假设集合(Hypothesis Set)，叫做 Perceptron Hypothesis 。**而 PLA 算法要做的，就是根据训练数据找到最优的w和threshold。

具体到上面有好货的实例当中可以看出：x={x1, x2}只有两维特征，代入h(x)中进行简化：
$$
h(x) = sign({ w_{1}}x_{1}+{ w_{2}}x_{2}-{ threshold})
$$
接着把所有的训练样本x1和x2绘制到二维平面上（根据方程w1x1+w2x2-threshold=0来绘制）：

![二维图像](/Users/randy/Documents/ml/machine_learning/resource/二维图像.jpg)

可以看到是一条直线，左边的点w1x1+w2x2-threshold<0，右边的点w1x1+w2x2-threshold>0，可以推出：

> PLA假设集合中任意一个确定的h(x)，都可视作一条直线将平面分隔成了两个区域。线的左边有h(x)=-1，右边有h(x)=1。

至此学习算法希望选中的模型就是上述的直线g，正好将训练数据划分为两个区域；可以知道该预测函数就是一个线性分类器。

事实上你会发现这样的直线存在很多条，那究竟如何选择一条最优的直线呢？也即确定最优的Wi和threshold的解。
$$
h(x) = sign(\sum_{i=1}^{n}({\color{Red} w_{i}}x_{i})-{\color{Red} threshold})
=sign(\sum_{i=1}^{n}({\color{Red} w_{i}}x_{i})-{\color{Red} (\overset{\underbrace{-threshlod}}{w_{0}})\times (\overset{\underbrace{+1}}{x_{0}})})=sign(\sum_{i=0}^{n}w_{i}x_{i})=sign(w^{T}x)
$$
这个变形目的是讲threshold统一收进w中去，这样PLA找到的最优的那条线就等价于找到了最优的参数w。

接下里就是用代码来实现PLA算法：

算法原理：

```tex
1、随便找一条线，即任意找一个n维向量w0，赋初值另w = w0
2、如果这条线正好把训练数据正确切分，Lucky！！训练结束！！此时w代表的h(x)就是我们学得的模型g
3、如果有任意一个样本没有被切分正确，即存在任意(x', y')，使得sign(w^T x') != y'，此时我们对w代表的线做一点点修正，另W_t+1 = w_t + y'x
4、跳转到Step 2
```

这里的关键就是步骤三的修正原理

![修正原理](/Users/randy/Documents/ml/machine_learning/resource/修正原理.png)

**以右上角的图形向量表示为例，w和x向量的夹角大于90度， 也就是说这两个向量的内积是负的，sign(wx)的结果就是-1，而我们需要的y则是+1，所以我们需要去修正这个内积的结果，使得它的符号是正的。只需要修正向量w的取值，保证新的w向量和x向量的夹角是小于90度的，这样我们对这个点的划分就是正确的。如上图所示我们将w更新为w+yx，这样根据四边形法则求内积结果符号就是正的。那这个更新公式是怎么来的呢？y = +1  w+xy=w+x 刚好将向量的夹角变小了，当然根据实际情况还可以添加系数t，令w=w+t* xy**

**注：两个向量的夹角大于90度，则向量内积就是负的，小于90度，向量内积是正的，可以直观的从图形上看出来**

预备数学知识：

```python
>>> sign = lambda x:1 if x > 0 else -1 if x < 0 else -1
>>> sign(2)
1
lambda表达式含义：lambda x:(1 if x > 0) else (-1 if x < 0) else -1
上述语句可以转换成：
def g(x):
  if(x > 0)
  	return 1
  else
  	if(x < 0)
    	return -1
    else
   		return -1
    
<> 表示不等于
```

```python
#coding:utf-8
from select import kevent

import numpy as np
import matplotlib.pyplot as plt

def draw(trainingData, w, round, x):
    plt.figure('Round'+str(round))
    drawLine(w)
    drawTrainingData(trainingData)

    if x is not None:
        plt.scatter(x[1], x[2], s= 400, c = 'red', marker=r'$\bigodot$')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def drawLine( w):
    w = w.transpose().tolist()[0]
    x1 = [x1 for x1 in xrange(80)]
    x2 = [ (w[0] + w[1] * i)/(-w[2]) for i in x1]
    plt.plot(x1, x2)


def drawTrainingData(trainingData):
    pointSize = 100
    positive_color = 'red'
    positive_marker = 'o'
    negative_clor = 'blue'
    negative_marker = 'x'

    positive_x1 = []
    positive_x2 = []
    negative_x1 = []
    negative_x2 = []
    for x, y in trainingData:
        x = x.transpose().tolist()[0]

        if y == 1:
            positive_x1.append(x[1])
            positive_x2.append(x[2])
        elif y == -1:
            negative_x1.append(x[1])
            negative_x2.append(x[2])

    plt.scatter(positive_x1, positive_x2, s= pointSize, c = positive_color, marker = positive_marker)
    plt.scatter(negative_x1, negative_x2, s= pointSize, c = negative_clor, marker=negative_marker)


def vector(l):
    return np.mat(l).transpose()

def trainingDataPreprocess(trainingData):
    '''Add x0 dimension & transform to np.mat object'''
    processedTrainingData = [ (vector([1, sample[0], sample[1]]), sample[2])  for sample in trainingData]
    return processedTrainingData

def PLA(trainingData):
    w = np.mat([1,2127,205]).transpose() # Step 1: 向量w赋初值

    k = 0 # 第k轮计数
    while True:
        k += 1

        (status, x, y) = noMistakePoint(trainingData, w)
        draw(trainingData, w, k, x) # 画图
        if status == 'YES': # Step 2: 切分正确，学习完成
            return w
        else:
            w = w + y*x # Step 3: 修正w

sign = lambda x:1 if x > 0 else -1 if x < 0 else -1
def mSign(m):
    '''判断某个矩阵的[0][0]元素正负.大于0返回1，否则返回-1'''
    x = m.tolist()[0][0]
    return 1 if x > 0 else -1 if x < 0 else -1

def noMistakePoint(training_data, w):
    '''训练数据中是否有点被切分错误'''
    status = 'YES'
    for (x, y) in training_data:
        if mSign(w.transpose() * x) <> sign(y):
            status = 'NO'
            return (status, x, y)

    return status, None, None

if __name__=="__main__":

    trainingData = [
    [10, 300, -1],
    [15, 377, -1],
    [50, 137, 1],
    [65, 92 , 1],
    [45, 528, -1],
    [61, 542, 1],
    [26, 394, -1],
    [37, 703, -1],
    [39, 244, 1],
    [41, 398, 1],
    [53, 495, 1],
    [32, 119, 1],
    [24, 577, -1],
    [56, 412, 1]
    ]

    processedTrainingData = trainingDataPreprocess(trainingData)
    w = PLA(processedTrainingData)
```

运行结果：

```python
w=  [[   1 2127  205]]
w=  [[   0 2117  -95]]
w=  [[   0 2117  -95]]
w=  [[   0 2117  -95]]
w=  [[   0 2117  -95]]
w=  [[   0 2117  -95]]
w=  [[  -1 2072 -623]]
w=  [[  -1 2072 -623]]
w=  [[  -1 2072 -623]]
w=  [[  -1 2072 -623]]
w=  [[  -1 2072 -623]]
w=  [[  -1 2072 -623]]
w=  [[   0 2133  -81]]
w=  [[   0 2133  -81]]
w=  [[  -1 2118 -458]]
w=  [[  -1 2118 -458]]
w=  [[  -1 2118 -458]]
w=  [[  -1 2118 -458]]
w=  [[  -1 2118 -458]]
w=  [[  -1 2118 -458]]
w=  [[   0 2179   84]]
w=  [[  -1 2169 -216]]
w=  [[  -1 2169 -216]]
w=  [[  -1 2169 -216]]
w=  [[  -1 2169 -216]]
w=  [[  -1 2169 -216]]
w=  [[  -1 2169 -216]]
w=  [[  -1 2169 -216]]
w=  [[  -1 2169 -216]]
w=  [[  -1 2169 -216]]
w=  [[  -1 2169 -216]]
w=  [[  -1 2169 -216]]
w=  [[  -1 2169 -216]]
w=  [[  -1 2169 -216]]
w=  [[  -1 2169 -216]]
```

PLA算法不一定会停下来，如果训练数据本身就不存在一条线可以将其正确切分，那么学习的过程将进入死循环。

> 训练数据是线性可分的，是PLA能够学习的必要条件



根据历史数据训练的模型对于未来预测是否可信呢？

计算学习理论（Computational Learning Theory）：搞清楚机器学习“为什么可以学习”





机器学习算法分类：

![机器学习算法分类](/Users/randy/machine_learning/machine_learning-action/machinelearninginaction/learn02/机器学习算法分类.png)

【监督学习】

> 通过外部的响应变量（Response Variable）来指导模型学习我们关心的任务，并达到我们需要的目的

目标：使模型可以更加准确的对我们所需要的响应变量建模

基础：三类模型（线性模型、决策树模型、神经网络模型）；此三类模型又可以细分为两类问题：分类问题和回归问题

分类问题的核心是如何利用模型来判别一个数据点的类别。这个类别一般是离散的，比如两类或者多类。回归问题的核心则是利用模型来输出一个预测的数值。这个数值一般是一个实数，是连续的

【无监督学习】

通常情况下，无监督学习并没有明显的响应变量，其核心往往是希望发现数据内部的潜在结构和规律，为我们进一步决断提供参考

典型的无监督学习就是希望利用数据特征来把数据分组（聚类）

无监督学习的另外一个作用是为监督学习提供更加有力的特征。通常情况下，无监督学习能够挖掘出数据内部的结构，而这些结构可能会比我们提供的数据特征更能抓住数据的本质联系，因此监督学习中往往也需要无监督学习来进行辅助。

工业级的人工智能产品：

最基本的概念就是：**你需要搭建一个管道让你的环境是动态的、闭环的**



分类方式：

1、按照输出y的值域空间进行分类

2、按照输出y的不同标签进行分类

3、按照模型训练的方式进行划分

开发机器学习应用程序的步骤：

1、收集数据

2、准备输入数据

3、分析输入数据

4、训练算法

5、测试算法

6、使用算法

基础的统计学和概率；线性代数以及多变量计算基础

Anaconda:用于科学计算的Python的发行版，提供包管理羽环境管理功能，方便解决多版本Python并存、切换及各种第三方包安装问题。Anaconda使用工具/命令conda来进行package和environment管理。

```latex
conda create --name my_app
```

添加Path：以zsh终端为例(普通终端采用.bashrc)

```cm
echo 'export PATH = "/Users/username/anaconda/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

conda的环境管理：

允许同时安装多个不同版本的Python，并能自由切换。对于上述安装假设采用Python2.7对应的安装包，则Python2.7就是默认环境（默认名为root）。接下来假如要安装Python3.4，要按照如下操作：

```latex
➜  ~ conda create --name python34 python=3.4
```

激活python3.4环境：

```latex
source activate python34  (for mac&linux)
activate python34  (for window)
```

此时再次输入	

```te
(python34) ➜  ~ python --version
Python 3.4.5 :: Continuum Analytics, Inc.
```

此时新安装的python3.4版本环境被放在了~/anaconda/envs目录下：

```tex
➜  envs ls
python34
```

也可以通过执行如下命令查看已安装环境

```te
➜  ~ conda info -e
# conda environments:
#
python34                 /Users/randy/anaconda/envs/python34
root                  *  /Users/randy/anaconda
```

取消激活，恢复为原来的环境

```te
source deactivate python34  (for mac&linux)
deactivate python34  (for window)
```

删除一个已有的环境

```te
conda remove --name python34 --all
```

conda的包管理：（这个跟pip很类似）

安装scipy

```te
conda install scipy
```

查看安装包列表：

```python
conda list
```

Mac上安装Anaconda之后，可以直接在GUI界面上launch jupyter notebook，或者直接命令行启动：

```tex
➜  ~ jupyter notebook
[I 20:11:30.053 NotebookApp] Serving notebooks from local directory: /Users/randy
[I 20:11:30.053 NotebookApp] 0 active kernels
[I 20:11:30.053 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888/?token=00ebaba29611bd0174437fd826b9105f0ec956539b0655bf
[I 20:11:30.053 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 20:11:30.054 NotebookApp]

    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=00ebaba29611bd0174437fd826b9105f0ec956539b0655bf
[I 20:11:30.513 NotebookApp] Accepting one-time-token-authenticated connection from ::1
或者
➜  ~ ipython notebook
```

然后浏览器就会打开localhost页面：http://localhost:8888/tree?



### 工程准备

1. 安装sklearn库：    

   ```py
   # 如果不想太麻烦，可以直接切换到根用户，这样就不用输sudo了
   sudo easy_install pip
   sudo pip install sklearn
   ```

2. 安装quandl

   ```pyh
   # 这个命令会安装numpy，pandas等库
   (sudo) pip install quandl
   pip install pandas
   ```

   问题1：在新mac上，会出现如下错误  *<u>OSError: [Errno 1] Operation not permitted:</u>*

   根据google查询得知新系统有个sip机制，默认下系统会启用该模式，即系统完整性保护机制，禁止对系统目录进行写操作。即使切换到root模式，也是不可行的。一种永久性做法就是：

   > 重启电脑，按住Command+R(直到出现苹果标志)进入Recovery Mode(恢复模式)
   > 左上角菜单里找到实用工具 -> 终端
   > 输入csrutil disable回车
   > 重启Mac即可

   另外一种比较优雅的做法就是：

   ```pyt
   pip install quandl --user -U
   ```

3. 安装ipython

   ```pyth
   pip install ipython --user -U
   ```

   问题2:安装之后直接在命令界面输入ipython会提示：<u>zsh: ipython: command not found</u>

   首先采用如下命令测试ipython是否真正安装上了：

   ```pyh
   python -m IPython
   ```

   ```tex
   ➜  Android-ItemTouchHelper-Demo git:(master) python -m IPython
   Python 2.7.10 (default, Jul 30 2016, 19:40:32)
   Type "copyright", "credits" or "license" for more information.

   IPython 5.3.0 -- An enhanced Interactive Python.
   ?         -> Introduction and overview of IPython's features.
   %quickref -> Quick reference.
   help      -> Python's own help system.
   object?   -> Details about 'object', use 'object??' for extra details.

   In [1]:
   ```

   可知确实已经安上，那就有可能是因为：`ipython`—the wrapper/launcher for it—is missing for whatever reason。所以这里可以采用别名的方式添加到shell startup脚本中：

   ```py
   alias ipython='python -m IPython'
   ```

   之后直接命令行输入ipython又可以了：

4. pandas:   Think of Pandas as a Python version of Excel. 

5. Scikit-learn, on the other hand, is an open-source machine learning library for Python.

## 实例演示

### 线性回归

接下来以有好货为例来说明线性回归是如何帮助我们预测成交额的？

![有好货用户成交额mock数据](/Users/randy/Documents/ml/有好货用户成交额mock数据.jpg)

通过上述训练数据我们希望得到一个可以预测用户消费额的模型。（x1, x2）=（年龄,购买力）样本标注 y = （花费费用）



接下来采用三步走的套路来分析线性回归模型的思路：

第一步，需要明确线性回归对现实场景是如何抽象的。顾名思义，线性回归认为现实场景中的响应变量（比如房价和票房）和数据特征之间存在线性关系。而线性会回归的数学假设有两个部分：

1、响应变量的预测值是数据特征的线性变换。这里的参数是一组系数。而预测值是系数和数据特征的线性组合。

2、

#### 采用google股票数据作为数据源，来进行线性预测（学习链接：https://www.youtube.com/watch?v=JcI5Vnw0b2c&index=2&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v）

这里采用的是Quandl中的stock数据

linear_regression.py

```python
import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGL')

print df.head()
```

Output:

```tex
➜  ~ python linear_regression.py
              Open    High     Low    Close      Volume  Ex-Dividend  \
Date                                                                   
2004-08-19  100.01  104.06   95.96  100.335  44659000.0          0.0   
2004-08-20  101.01  109.08  100.50  108.310  22834300.0          0.0   
2004-08-23  110.76  113.48  109.05  109.400  18256100.0          0.0   
2004-08-24  111.24  111.60  103.57  104.870  15247300.0          0.0   
2004-08-25  104.76  108.00  103.88  106.000   9188600.0          0.0   

            Split Ratio  Adj. Open  Adj. High   Adj. Low  Adj. Close  \
Date                                                                   
2004-08-19          1.0  50.159839  52.191109  48.128568   50.322842   
2004-08-20          1.0  50.661387  54.708881  50.405597   54.322689   
2004-08-23          1.0  55.551482  56.915693  54.693835   54.869377   
2004-08-24          1.0  55.792225  55.972783  51.945350   52.597363   
2004-08-25          1.0  52.542193  54.167209  52.100830   53.164113   

            Adj. Volume  
Date                     
2004-08-19   44659000.0  
2004-08-20   22834300.0  
2004-08-23   18256100.0  
2004-08-24   15247300.0  
2004-08-25    9188600.0  
```

从上述数据输出结果可知，该数据包含多个特征（feature），比如Open，High，Low等

数据的每一列对应一个feature，在机器学习中我们可以获取所需的所有特征，但是我们只关心那些有意义的features。线性回归在这里没法直接找出这些特征之间的关系，有些特征虽然没有什么价值，但是一旦联系与其他特征之间的联系，就可能得到一些有意义的结果。比如股票走向的谷峰和谷底的差值可以说明股票在一天之类的上升或下降的幅度等。

接下来对特征进行一些处理，同时定义一些新的特征，下面的示例只关心了部分features：(代码的具体含义待说明TODO)

```python
import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Open'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

print df.head()
```
   Output：

   ```tex
   ➜  ~ python linear_regression.py
                  Adj. Close    HL_PCT  PCT_change  Adj. Volume
      Date                                                     
      2004-08-19   50.322842  3.724628    0.324968   44659000.0
      2004-08-20   54.322689  0.762301    7.227007   22834300.0
      2004-08-23   54.869377  3.683640   -1.227880   18256100.0
      2004-08-24   52.597363  6.049982   -5.726357   15247300.0
      2004-08-25   53.164113  1.909126    1.183658    9188600.0
   ```

单一变量的线性回归：

逻辑回归：


2. 连续监督学习（回归）

   ​

3. ​

https://medium.com/@suffiyanz/getting-started-with-machine-learning-f15df1c283ea

4、



### Kaggle

采用pandas操作数据：laoding data and cleaning data；使用scikit-learn做出预测；

参考：https://github.com/savarin/pyconuk-introtutorial

问题1：泰坦尼克号沉船预测 https://www.kaggle.com/c/titanic

###集体智慧编程

####协作型过滤

相似度评价值计算体系：欧几里得距离和皮尔逊相关度

皮尔逊相关度计算：该系数是用于判断l两组数据与某一直线拟合程度的一种度量方式。【适用于数据不是很规范，比如影评人对影片的评价总是相对于平均水平偏离很大】

> 注：皮尔逊相关度计算进行评价，它修正了“夸大分值”的情况，其绘制原则是尽可能地靠近图上所有坐标点，寻求一个最佳拟合曲线。如果某个人总是倾向于给出比另一个人更高的分值，而两者的分值之差又始终保持一致，则他们依然会存在很好的相关性。这一点也是区别于欧几里得计算方式的地方。

安装pydelicious:  python setup.py install
—>安装目录  /Users/randy/anaconda/lib/python2.7/site-packages/feedparser-5.2.1-py2.7.egg

附：RSS订阅源是一个包含博客及其所有文章条目信息的简单的XML文档【RSS解析：[Feedparser](https://pythonhosted.org/feedparser/)】





### 机器学习常用算法

#### 聚类

##### 应用

应用场景：数据量很大的应用

#### K-近邻算法 KNN（k-means）

> 优点：精度高、对异常值不敏感、无数据输入假定
>
> 缺点：计算复杂度高、空间复杂度高
>
> 适用数据范围：数值型和标称型

#####概念

训练样本集:{ 样本1,  样本2, … , 样本n}

每个样本对应一个标签（即明确样本集中每一数据与所属分类的对应关系）

#####预测

输入：没有标签的新数据——>比较新数据和样本集数据特征——>算法提取样本集中特征最相似（最近邻）数据的分类标签

一般取样本集中最相似的前k个最相似的数据（通常k不大于20） ——> 选择k个最相似数据张出现次数最多的分类作为新数据分类

#####示例

分辨一部电影是爱情片还是动作片？

步骤：

1、计算位置电影与样本集中其他电影的距离

计算方法：

计算结果：

California Man         				20.5

He's Not Really into Dudes    		18.7

Beautiful Woman 				19.2

Kevin Longable					115.3

Robo Slayer 3000					117.4

Amped II						118.9

取k=3，可知距离最近的散步电影全是爱情片，所以判定位置电影为爱情片

#####K-近邻算法的一般流程

1、收集数据：可以使用任何方法，比如已有的文本文件

2、准备数据：距离计算所需要的数值，最好是结构化的数据格式（比如可以用python解析文本文件）

3、分析数据：可以使用任何方法（比如，用Matplotlib绘制二维扩散图）

4、训练算法：此步骤不适用于k-近邻算法

5、测试算法：计算错误率

6、使用算法：首先需要输入样本数据和结构化的输出结果，然后运行k-近邻算法判定输入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理



KNN分类算法：

对每组数据进行分类：

1、计算已知类别数据集中的点与当前点之间的距离

2、按照距离递增次序排序

3、选取与当前点距离最小的k个点

4、确定前k个点所在类别出现的频率

5、返回前k个点出现频率最高的类别作为当前点的预测分类

代码清单：learn01/knn.py  —   classify

第一个分类器：

```python
#coding:utf-8
from numpy import *
import operator

# 这个样例只是用于熟悉整个knn算法执行的流程，并没有实际意义

def createDataSet():
	# group矩阵中含有4组数据，每组数据包含两个已知的属性
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	# 向量labels包含每个数据点的标签信息，元素个数等于group矩阵的行数，比如数据点（1.0,0.1）定义成类A
	labels = ['A', 'A', 'B', 'B']
	return group, labels


def classify(inX, dataSet, labels, k):
	# 数据集包含四行，也即对应四个标签
	dataSetSize = dataSet.shape[0]
	# 对inX在行方向上重复dataSetSize次，在列方向上重复1次，然后和dataSet相减
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	#
	sqDiffMat = diffMat ** 2
	# 平方求和
	sqDistances = sqDiffMat.sum(axis=1)
	# 开平方
	distances = sqDistances ** 0.5

	sortedDistIndices = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndices[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	sortedClassCount = sorted(classCount.iteritems(),
		key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

if __name__ == '__main__':
	group, labels = createDataSet()
	result = classify([0, 0], group, labels, 3)
	print result
```

运行结果为： B

**问题：上述分类器的结果是否总是正确呢？答案是否定的，那该如何去检测分类器的正确率呢？**

一种测试方式就是使用已知答案的数据，让分类器去预测，然后比较输出结果是否符合预期



注：收集和准备数据的过程实际上就是一种数据清洗的过程，将一些无效的脏数据给剔出去，保留有效数据



第二个分类器：改进约会网站的配对效果

三种类型：

不喜欢的人   			1

魅力一般的人   		2

极具魅力的人			3

首先准备数据：从收集的文本文件中解析数据

样本包含三种特征：

1、每年获得的飞行常客里程数

2、玩视频游戏所消耗的事件百分比

3、每周消费的冰淇淋公升数

**数据格式处理：即输入为文件名字字符串，输出为训练样本矩阵和类标签向量**

```python
def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	# 返回一个numberOfLines行，3列的numpy零矩阵
	returnMat = zeros((numberOfLines, 3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		# 移除字符串头尾指定的字符（默认为空格），截掉回车字符
		line = line.strip()
		# 使用tab字符将上一步得到的整行数据分割成一个元素列表
		listFromLine = line.split('\t')
		# 选取前3个元素，将他们存储到特征矩阵中
		returnMat[index, :] = listFromLine[0:3]
		# python中使用索引值-1表示列表中的最后一列元素（即将最后一列的标签分类存储到向量中）
		# 这里需明确将数据指定为整型，python默认将这些元素当字符串处理
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector


if __name__ == '__main__':
	group, labels = createDataSet()
	result = classify([0, 0], group, labels, 3)
	print result

	datingDataMat, datingfLabels = file2matrix('datingTestSet.txt')
	print datingDataMat
	print datingfLabels[0:20]
```

运行结果：

```tex
➜  learn02 git:(master) ✗ python knn.py
B
[[  4.09200000e+04   8.32697600e+00   9.53952000e-01]
 [  1.44880000e+04   7.15346900e+00   1.67390400e+00]
 [  2.60520000e+04   1.44187100e+00   8.05124000e-01]
 ...,
 [  2.65750000e+04   1.06501020e+01   8.66627000e-01]
 [  4.81110000e+04   9.13452800e+00   7.28045000e-01]
 [  4.37570000e+04   7.88260100e+00   1.33244600e+00]]
[3, 2, 1, 1, 1, 1, 3, 3, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3]
```

至此已经将我们需要的数据格式化了，接下来为了更加直观的展示数据，将采用图形化的方式呈现：

```python
def matrix2graph(dataMat):
	fig = plt.figure()
	ax = fig.add_subplot(111)
    # 第一列作为横坐标(玩视频游戏所耗时间百分比) 第二列作为纵坐标(每周消费的冰淇淋公升数)
	ax.scatter(dataMat[:, 1], dataMat[:, 2])
	plt.show()

if __name__ == '__main__':
	group, labels = createDataSet()
	result = classify([0, 0], group, labels, 3)
	print result

	datingDataMat, datingfLabels = file2matrix('datingTestSet.txt')
	print datingDataMat
	print datingfLabels[0:20]

	# draw matrix
	matrix2graph(datingDataMat)
```

运行结果：

![figure_1](/Users/randy/machine_learning/machine_learning-action/machinelearninginaction/learn02/figure_1.png)

从上图我们会发现由于没有采用样本分类的特征值，很难看出任何有用的数据模式信息。所以需要个性化标记：

调整代码：

```python
ax.scatter(dataMat[:, 1], dataMat[:, 2], 15.0*array(datingfLabels), 15.0*array(datingfLabels))
```

再次运行：

![figure_2](/Users/randy/machine_learning/machine_learning-action/machinelearninginaction/learn02/figure_2.png)

这样根据标签属性绘制不同的颜色点，就很容易区分出数据点所属三个样本的区域轮廓。

在提取到我们所需要的格式的样本数据之后，接下来我们将要进一步对数据做处理，即进入准备数据的过程：

**对数据进行归一化处理：**

原因：我们在计算样本1和样本2数据之间的距离时，一般采用三个特征值差的平方求和再开方的方式来计算，这里就存在一个问题就是，差值最大的属性对于计算结果的影响最大，在上述例子中，每年获取的飞行常客里程数对于结果的影响明显高于其他两个属性，而对于最终的预测结果来说，三个属性是等权重的，所以需要进行归一化处理。

即将数值处理为0到1或者-1到1之间，可采用如下公式：
$$
newValue = (oldValue-min) / (max-min)
$$
其中max和min分别表示数据集中最大和最小的特征值。接下来看代码如何实现归一化：

```python
def autoNorm(dataSet):
	# 取矩阵每一列的最小值和最大值
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	# 按照dataSet的维度取一个零矩阵
	normDataSet = zeros(shape(dataSet))
	# 取多维数组的行数
	m = dataSet.shape[0]
	# tile对minVales在行方向上重复m次，列方向上重复1次
	normDataSet = dataSet - tile(minVals, (m, 1))
	normDataSet = normDataSet/tile(ranges, (m, 1))
	return normDataSet, ranges, minVals
```

这里注意：特征值矩阵有1000X3个值，而minVals和maxVals都是1X3。为了解决这个问题，采用tile函数将变量内容扩展为1000行；此外这里的除法/是表示特征值相除，而非矩阵相除，在numpy中矩阵相除是采用函数linalg.solve(matA, matB)的。

接下来测试该函数：

```python
normMat, ranges, minVals = autoNorm(datingDataMat)
print normMat
print ranges
print minVals
```

输出结果：

```tex
[[ 0.44832535  0.39805139  0.56233353]
 [ 0.15873259  0.34195467  0.98724416]
 [ 0.28542943  0.06892523  0.47449629]
 ...,
 [ 0.29115949  0.50910294  0.51079493]
 [ 0.52711097  0.43665451  0.4290048 ]
 [ 0.47940793  0.3768091   0.78571804]]
[  9.12730000e+04   2.09193490e+01   1.69436100e+00]
[ 0.        0.        0.001156]
```

**算法测试**

机器学习一个很重要的工作就是评估算法的正确率，通常我们只提供已有数据的90%作为样本数据，剩下的10%数据去测试分类器，检测分类器的正确率。

10%的数据随机选择，代码里可以直接采用计数器来累加错误的次数，最后除以总的测试次数就得到了错误率

测试代码：

```python
def datingClassTest():
	hoRatio = 0.10      #hold out 10%
	datingDataMat,datingLabels = file2matrix('datingTestSet.txt')       #load data setfrom file
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
	    classifierResult = classify(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
	    print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
	    if (classifierResult != datingLabels[i]): errorCount += 1.0
	print "the total error rate is: %f" % (errorCount/float(numTestVecs))
	print errorCount
```

运行结果：

```tex
...
the classifier came back with: 1, the real answer is: 1
the classifier came back with: 1, the real answer is: 1
the classifier came back with: 3, the real answer is: 3
the classifier came back with: 2, the real answer is: 3
the classifier came back with: 1, the real answer is: 1
the classifier came back with: 2, the real answer is: 2
the classifier came back with: 1, the real answer is: 1
the classifier came back with: 3, the real answer is: 3
the classifier came back with: 3, the real answer is: 3
the classifier came back with: 2, the real answer is: 2
the classifier came back with: 1, the real answer is: 1
the classifier came back with: 3, the real answer is: 1
the total error rate is: 0.050000
5.0
```

错误率为5%，我们可以通过改变hoRatio和变量k的值，观察错误率的变化情况，依赖于分类算法，数据集和程序设置，分类器的输出结果会有很大不同

> 对此，如何去衡量，量化算法的优化方向，依据是什么？

第三个分类器：手写识别系统

为简化起见，这里的构造系统只能识别数字0和9，为了方便测试和理解，这里采取将图像转换成文本格式：

```tex
00000000000001100000000000000000
00000000000011111100000000000000
00000000000111111111000000000000
00000000011111111111000000000000
00000001111111111111100000000000
00000000111111100011110000000000
00000001111110000001110000000000
00000001111110000001110000000000
00000011111100000001110000000000
00000011111100000001111000000000
00000011111100000000011100000000
00000011111100000000011100000000
00000011111000000000001110000000
00000011111000000000001110000000
00000001111100000000000111000000
00000001111100000000000111000000
00000001111100000000000111000000
00000011111000000000000111000000
00000011111000000000000111000000
00000000111100000000000011100000
00000000111100000000000111100000
00000000111100000000000111100000
00000000111100000000001111100000
00000000011110000000000111110000
00000000011111000000001111100000
00000000011111000000011111100000
00000000011111000000111111000000
00000000011111100011111111000000
00000000000111111111111110000000
00000000000111111111111100000000
00000000000011111111110000000000
00000000000000111110000000000000
```

通过1和0描述数字0，其他数字类似处理，接下来采用两个目录存储图像文本，一个是测试数据集，一个是样本集。

接下来采用代码将32X32的二进制表示写入到1X1024的向量中：

```python
# 采用一行一行的连续添加到returnVect数组中,即将图像文本的32X32的01表示重写改写成1X1024的数组形式
def img2vector(filename):
	returnVect = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		# print "lineStr: " + lineStr
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect
```

```python
....
testVector = img2vector('testDigits/0_13.txt')
print testVector[0, 0:31]
```



运行输出，与原文本对比：

```
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  1.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
```

对比可知无误。

接下来就是采用上面的k-近邻算法进行测试：

注意文件命名规则：比如9_45.txt表示分类是9， 后面的45表示分类9的第45个实例

接下来我们在knn.py中写一个自包含的函数用于测试：

```python
handWritingClassTest():
def handWritingClassTest():
	hwLabels = []
	# 获取目录内容
	trainingFileList = listdir('trainingDigits')
	# print "file list: " + trainingFileList
	m = len(trainingFileList)
	# 每个文本对应一行，共1024列，多少个文本，就对应矩阵的多少行
	trainingMat = zeros((m, 1024))
	# 文件名命名是和文本中的数字对应的，接下来根据文件名解析分类数字
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		# 获取分类
		classNumStr = int(fileStr.split('_')[0])
		# 将分类存储在hwLabels向量中
		hwLabels.append(classNumStr)
		# 载入图像，对数据进行格式化处理
		trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		# 上述操作类似，接下来不存入矩阵，而是调用分类函数，对测试文件夹中的文件分别进行测试属于哪个分类
		classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
		# 注意这里由于文本中的数值本身就在0和1之间，所以不需要调用autoNorm()函数了
		print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
		# 判断分类结果和实际的分类是否一致，计算错误率
		if (classifierResult != classNumStr):
			errorCount += 1.0
	print "\nthe total number of errors is: %d" % errorCount
	print "\nthe total error rate id: %f" % (errorCount/float(mTest))
```

运行结果：（结果太长，只截取最后一段）

```tex
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9
the classifier came back with: 9, the real answer is: 9

the total number of errors is: 11

the total error rate id: 0.011628
```

可知错误率为1.2%，改变变量k的值，修改函数handWritingClassTest函数随机选取训练样本，改变训练样本数目，都会对k-近邻算法的错误率产生影响

TODO:改变相应参数，观察错误率变化

实际运行过程及代码分析可知，算法执行效率并不高。因为算法需要对每个测试向量做2000次距离计算，每个距离计算包含了1024个维度浮点运算，总计要执行900次，此外还需要为测试准备2MB的存储空间，所以后续会采用改进算法：**k决策树**

总结：k-近邻算法是分类数据最简单最有效的算法。前面分析也指出该算法比较耗时，而且需要较大的存储空间，除此之外，还有一个缺陷就是无法给出任何数据的基础结构信息，因此无法知道平均实例样本和典型实例样本具有什么特征。

关于k-近邻算法主要存在三个关键问题：

1、k值的选择：k是唯一的参数，参数选取一般用交叉验证，如v折交叉验证，留一法等。

2、距离度量方法：上文中采用的是欧式距离，此外还有曼哈顿距离、汉明距离等

3、特征选择：上文中约会的样例中是依据约会者主观的的考察属性，后面的数字识别没有提到特征选取

参考博客：https://my.oschina.net/u/1412321/blog/194174

#### 决策树（Decision Trees）

##### 概念

> 基于树结构进行决策。简单来说就是类似二分任务，作分支判断

决策过程中提出的每个判定问题都是对某个属性的测试，每个测试结果或导出最终结论，或是导出进一步的判定问题，其考虑范围是在上次决策结果的限定范围之类。

##### 决策数的结构特征

1、一般的，一棵决策树包含一个根节点、若干个内部节点和若干个叶子节点；

2、叶子节点对应于决策结果，其他每个节点则对应于一个属性测试；

3、每个节点包含的样本集合根据属性测试的结果被划分到子节点中

4、根节点包含样本全集

##### 决策树的学习目的

> 为了产生一棵泛化能力强，即处理未见示例能力强的决策树，其基本流程遵循简单且直观的”分而治之“策略

##### 决策树基本算法

决策树的生成是一个递归的过程，在该算法中会存在三种情形导致递归返回：
1、当前节点包含的样本属于同一分类，无需划分

2、当前属性集为空，或者所有样本在所有属性上取值相同，无法划分

3、当前节点包含的样本集合为空，不能划分

##### 优点

计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据

##### 缺点

可能会产生过度匹配问题

##### 适用数据类型

数值型和标称型



前面讲到了k-近邻算法，分析了该算法的最大缺点就是无法给出数据的内在含义，而决策树的主要优势就在于数据形式非常容易理解。接下来采用流程图的形式描述一下决策树的概念：



决策树的一个重要任务就是为了理解数据蕴含的知识信息，因此决策树可以利用不熟悉的数据集合，并从中提取出一系列规则，这些机器根据数据集创建规则的过程就是机器学习的过程。

##### 实践方法

1. 从一堆原始数据中构造决策树。
2. 剔除一些度量算法成功率的方法
3. 使用递归建立分类器并且使用MatplotLib绘制决策树图

采用的样例是输入隐形眼镜的处方数据，并由决策树分类器预测需要的镜片类型

第一步是关键，要讨论数学上如何使用信息论划分数据集，然后编码将理论运用到具体数据集中，进而构建决策树。

解决的第一个问题：找出数据集中哪个特征在划分数据分类时起决定性作用。

**信息增益**

划分数据集的大原则是：将无序的数据变得更加有序。信息熵是度量样本集合纯度最常用的一种指标。

信息熵的定义：假定当前样本集合D中第k类样本所占的比例为
$$
{p}_{k}(k = 1,2,...,|Y|)
$$
，则D的信息熵定义为：
$$
Ent\left(D \right) = -\sum_{k=1}^{|y|}{p}_{k}\log_ {2}{p}_{k}
$$
Ent(D)的值越小，则D的纯度越高。

信息增益公式：
$$
Gain(D, a) = Ent(D) -\sum_{v=1}^{V}\frac{|{D}^{v}|}{|D|}Ent({D}^{v})
$$
一般而言，信息增益越大，则意味着使用属性a来进行划分所获得的”纯度提升“越大。因此我们可以采用信息增益来进行决策树的划分属性选择。也即从属性集中选择最优划分属性：
$$
{a}_{*} = \arg \max Gain(D, a)  (a \epsilon  A)
$$
注：机器学习入门视频笔记

利用决策树一个一个的处理多元线性问题

下面结合具体的示例来学习如何使用python计算信息熵：

​      		不浮出水面是否可以生存     		是否有脚蹼				属于鱼类

1				是						是						是

2				是						是						是

3				是						否						否

4				否						是						否

5				否						是						否

该示例目标是学习该海洋生物是否是鱼类。显可以知道|Y| = 2。在决策树学习开始时，根节点包含D中的所有样例，其中正例占p1=2/5，反例占p2=3/5。于是根据上述公式可以计算出根节点的信息熵为：
$$
Ent(D) = -\frac{2}{5}\log_ {2}{\frac{2}{5}}-\frac{3}{5}\log_ {2}{\frac{3}{5}} = 0.971
$$
当前的属性集合是{不浮出水面是否可以生存，是否有脚蹼}，以”不浮出水面是否可以生存“属性为例，若使用该属性对D（即是否为鱼类）进行划分，可以得到两个子集：D1 = {不浮出水面是否可以生存 | 是}， D2 = {不浮出水面是否可以生存 | 否}。子集D1包含编号{1， 2， 3}的三个样例，其中正例p1 = 2/3， p2 = 1/3。由此计算出根据”不浮出水面是否可以生存“属性划分是否为鱼类的2个分支节点的信息熵是：
$$
Ent({D}^{1}) = -\frac{2}{3}\log_ {2}{\frac{2}{3}}-\frac{1}{3}\log_ {2}{\frac{1}{3}} =
$$

$$
Ent({D}^{2}) = -0-\frac{2}{2}\log_ {2}{\frac{2}{2}} = 0
$$

可以知道不浮出水面不能生存的生物一定不是鱼类，这个分支就终结了。

```python
#coding:utf-8
from numpy import *
from math import log

# 计算给定数据集的熵
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	print "data length: " + bytes(numEntries)
	labelCounts = {}
	# 遍历dataSet矩阵的每一行
	for featVec in dataSet:
		# 获取每一行的最后一个元素
		currentLabel = featVec[-1]
		print "current label: " + currentLabel
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1

	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt

def createDataSet():
	dataSet = [	[1, 1, 'yes'],
				[1, 1, 'yes'],
				[1, 0, 'no'],
				[0, 1, 'no'],
				[0, 1, 'no']
			   ]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

if __name__ == '__main__':
	myData, labels = createDataSet()
	print myData

	result = calcShannonEnt(myData)
	print result
```

实际计算公式就是：
$$
-\frac{2}{5}\log_ {2}{\frac{2}{5}}-\frac{3}{5}\log_ {2}{\frac{3}{5}}
$$
运行结果如下：

```tex
➜  learn03 git:(master) ✗ python trees.py
[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
data length: 5
current label: yes
current label: yes
current label: no
current label: no
current label: no
0.970950594455
```

熵越高，则混合的数据也越多，我们可以在数据集中增加更多的分类，观察熵是如何变化的。下面增加一个名为randy的分类，测试熵的变化：

```python
dataSet = [	[1, 1, 'maybe'],
           [1, 1, 'yes'],
           [1, 1, 'yes'],
           [1, 0, 'no'],
           [0, 1, 'no'],
           [0, 1, 'no']
		 ]
```

测试输出熵的结果为：1.45914791703

在获取了熵之后，我们接下来就可以按照获取最大信息增益的方法划分数据集。

接下来就是计算当前属性集合中每个属性的信息增益，对于上例就是两个属性{不浮出水面是否可以生存，是否有脚蹼}，为了简化后面的描述，分别用字母A和B表示这两个属性。接下来以计算属性A的信息增益为例说明：

属性A的取值只有两种：是或否，根据信息熵的计算方式，可得
$$
Ent({D}^{1}) = -\frac{2}{3}\log_ {2}{\frac{2}{3}}-\frac{1}{3}\log_ {2}{\frac{1}{3}} = 2.11
$$

$$
Ent({D}^{2}) = -0-\frac{2}{2}\log_ {2}{\frac{2}{2}} = 0
$$

计算信息增益：
$$
0.971-(2.11 * 3/5 + 0 * 2/2) = 
$$

#### 支持向量机（SVM）

##### 概念

> SVM是一系列可用于分类、回归和异常值检测的有监督学习方法

##### 优点

1. 在高维空间行之有效
2. 当维数大于样本数时仍然可用
3. 在决策函数中只使用训练点的一个子集（称为支持向量），大大节省了内存开销
4. 用途广泛：决策函数中可以使用不同的[核函数](http://scikit-learn.org/stable/modules/svm.html#svm-kernels)。提供了一种通用的核，但是也可以指定自定义的核

##### 缺点

1. 如果特征数量远大于样本数量，则表现会比较差
2. SVM不直接提供概率估计。这个值通过五折交叉验证计算，代价比较高


##### 适用数据类型

数值型和标称型数据

简述：通俗的讲，SVM是一种二类分类模型，其基本模型定义为特征空间上的间隔最大的线性分类器，即支持向量机的学习策略便是间隔最大化，最终可转化为一个凸二次规划问题的求解。或者简单的可以理解为就是在高维空间中寻找一个合理的超平面将数据点分隔开来，其中涉及到非线性数据到高维的映射以达到数据线性可分的目的。




选取分割线：以两个分类为例，最佳的分割线最大化了到最近点的距离，并且对涉及的两个分类最大化了此类距离。该距离通常称为间隔（Margin）

支持向量机的内部原理就是最大限度地提升结果的稳健性，对未见示例具有最强的泛化能力

注意：SVM总是将正确分类标签作为首要考虑，然后对间隔进行最大化

因此，支持向量就是离分隔超平面最近的那些点。



支持向量的概念：

![svm_linear](/Users/randy/Documents/ml/machine_learning/resource/svm_linear.png)

上面样本图是一个特殊的二维情况，真实情况当然可能是很多维。先从低纬度简单理解一下什么是支持向量。从图中可以看到3条线，中间那条红色的线到其他两条先的距离相等。这条红色的就是SVM在二维情况下要寻找的超平面，用于二分类数据。而支撑另外两条线上的点就是所谓的支持向量。从图中可以看到，中间的超平面和另外两条线中间是没有样本的。找到这个超平面后，利用超平面的数据数学表示来对样本数据进行二分类，就是SVM的机制了。

几个概念：

- (1)如果能找到一个直线（或多维的面）将样本点分开，那么这组数据就是线性可分的。将上述数据集分隔开来的直线(或多维的面)称为分隔超平面。分布在超平面一侧的数据属于一个类别，分布在超平面另一侧的数据属于另一个类别
- (2)支持向量（Support vector）就是分离超平面最近的那些点。
- (3)几乎所有分类问题都可以使用SVM，值得一提的是，SVM本身是一个二分类分类器，对多类问题应用SVM需要对代码做一些修改。

示例：

```python
#coding:utf-8
from sklearn import svm
# 向量点  [1, 1], [2, 3]两个点分别位于超平面两边的支持向量上
x = [[2, 0], [1, 1], [2, 3]]
# label
y = [0, 0, 1]
# 调用线性核函数
clf = svm.SVC(kernel='linear')
clf.fit(x, y)
print clf
# get support vectors  获取支持向量机(即输出支持向量上的点)
print clf.support_vectors_
# get indices of support vectors  获取支持向量的下标
print clf.support_  
# get number of support vectors for each class 获取每个分类标签下的支持向量的个数
print clf.n_support_
```

运行结果：

```tex
➜  svm git:(master) ✗ python SklearnSVM.py
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
[[ 1.  1.]
 [ 2.  3.]]
[1 2]
[1 1]
```

有时很明显不存在将两个类分割的决策面，可以将某些点看作是异常值，对此SVM是如何应对的呢？

SVM对于异常值是比较健壮的，这在某种程度上均衡了它找出最大距离的间隔和忽略异常值的能力。

https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650728330&idx=2&sn=dadc2ac166249dbb600014c44077ae87

对于支持向量机，最好的超平面是最大化两个类别边距的那种方式，换言之，超平面对每个类别最近的元素距离最远。（The best choice will be the hyperplane that leaves the maximum margin from both classes）

如何建造超平面？

线性可区分（linear separable）和线性不可区分（linear inseparable）

- 线性可区分

  - 定义和公式的建立：

  - 超平面定义：
    $$
    \vec{W} \cdot \vec{X} + b=0
    $$










$$
\vec{W}: weight\ vector   \\   \vec{W} = \{{w}_{1}, {w}_{2}, ...,{w}_{n}\}
$$

n表示特征值的个数；X训练实例（n x 1的维度）；b表示bias

对于简单的情形，我们可以找到一个线性决策边界（即一条直线分开两个类别）；但是有些复杂的情形是无法找出这样一条直线的，有时需要引入第三个维度，即三维空间（x，y，z）（这时就需要用到核函数，将数据从低纬度映射到高纬度，使其线性可分，然后再应用SVM理论）

![xy_2](/Users/randy/Documents/ml/machine_learning/resource/xy_2.png)

由上图可知，在这个二维平面上是没有办法找到一条线性的超平面来将两个特征分割开的，这时我们引入第三个维度，即z轴，根据三维空间的概念，我们可以将三角形的特征值沿着z轴向上移动，这样从z轴这个维度上来看就将三角形和圆形两中特征分成了上下两部分，这样以z轴的某个刻度做一个平行于x轴的平面就可以很轻松的将两种特征分开。由z轴向xy轴组成的二维平面投影如下图所示：z = x² + y²

![xyz](/Users/randy/Documents/ml/machine_learning/resource/xyz.png)

于是，我们的决策边界就成了半径为1的圆形。

上述例子中我们找到一种通过将空间巧妙的映射到更高维度来分类非线性数据的方法。然而事实证明，这种转换可能会带来很大的计算成本：可能会出现很多新的维度，而每一个维度都可能带来复杂的计算。所以需要找到一种更简单的方式。

核函数：

​				
​					∆wj =η(y(i)−yˆ(i))x(i)

####朴素贝叶斯分类

使用朴素贝叶斯分类进行文档分类：
观察文档中出现的词，并把每个词的出现和不出现作为一个特征

首先要从文本中获取特征，那就需要先拆分文本。这里的特征来自于文本的词条（token），一个词条是字符的任意组合。然后将每一个文本片段表示成一个词条向量，其中值1表示词条出现在文档中，值0表示词条未出现。



















