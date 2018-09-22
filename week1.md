# Machine-learning-note
Recording machine learning perceptions

# 前言：
机器学习是目前最激动人心的技术之一，你也许没有意识到自己每天都会多次用到学习算法，每当你使用Google或百度等搜索引擎时，它总是能给出令人满意的结果，原因就是Google或百度等公司使用的学习算法，学会了如何给网页排序。每次你使用Facebook或苹果的相片分类功能，它能识别出你朋友的照片，这也是机器学习。每当你阅读邮件时，你的垃圾邮件过滤器会帮助你过滤掉大量的垃圾邮件，这也是学习算法。

那么为什么机器学习现在如此流行？实际上机器学习是从AI即人工智能发展出来的一个领域。我们想建造智能机器，然后发现我们可以通过编程让机器做一些基本的事情。比如，如何找到从A到B的最短路径。但是大多数情况下，我们不知道如何编写AI程序来做更有趣的事情。如网页搜索，相片标记，反垃圾邮件，人们认识到做到这些事情唯一的方法，就是使机器自己学习如何去做。因此，机器学习是为计算机开发的一项新功能，如今它涉及工业和基础科学中的许多领域。例如数据挖掘，我们无法手动编写的程序（例如无法编写程序使直升飞机自己飞行，唯一可行的就是让计算机自己学习驾驶直升飞机）。

## 1.机器学习介绍

### 1.1 什么是机器学习（Machine Learning）?

即使是在机器学习从业者中，也没有对机器学习的统一定义，一个比较早期的定义是由Arthur Samuel在1959年给出的，他将机器学习定义为：**在没有明确设置的情况下，使计算机具有学习能力的研究领域。**这是一个不正式，也是比较陈旧的一个定义。

Samuel编写了一个跳棋游戏的程序,这个跳棋游戏令人惊讶之处在于，Samuel自己并不是一个玩跳棋的高手，他所做的是使程序与自己对弈几万次，通过观察哪些布局容易赢,哪些布局容易输，一段时间后，跳棋游戏程序就学到了什么是好的布局，什么是不好的布局，最终程序学会了玩跳棋，比Samuel玩得还好。

Tom Mitchell在1998年提出了一个更新的定义：**计算机程序从经验E中学习，解决某一任务T，进行某一性能度量P，通过P测定在T上的表现因经验E而提高。对于上述的跳棋游戏，经验E就是程序与自己下几万次跳棋，任务T就是玩跳棋，性能度量P就是与新对手玩跳棋时赢的概率**。

### 1.2 机器学习分类：

目前有各种不同类型的学习算法，最主要的两类是**监督学习**(supervised learning)和**无监督学习**(unsupervised learning)，通俗来讲，监督学习就是我们会教计算机做某件事情。而在无监督学习中，我们让计算机自己学习。

#### Supervised Learning：

在给出监督学习的定义前，先来看一个例子

假设你拥有下面的这些房价数据，图表上的每一个点代表一次交易，横坐标代表交易房屋的占地面积，纵坐标代表交易房屋的价格（单位是千美元）。

![](https://github.com/17edelweiss/Machine-learning-note/blob/master/File1/1.png?raw=true)

现在让你预测一个750平方英尺的房屋的交易价格可能是多少？学习算法能做到的一件事就是根据数据画一条直线来拟合这些数据，然后根据这条直线来进行预测。但是这不是能使用的唯一的学习算法，在此例子中，可能会有一个更好的学习算法，例如用二次函数或二阶多项式看起来能够模拟得更好。

![https://img-blog.csdn.net/201809141507276?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

监督学习是指，我们给算法一个数据集，其中包含了正确答案，也就是说我们给它一个房价数据集，在这个数据集中的每个样本，我们都给出正确的价格，即这个房子的实际卖价，算法的目的就是给出更多的正确答案，例如通过算法预测房价为750平方英尺的房子的价格。用更专业的术语来定义，这个问题也被称为回归问题，回归问题就是指**预测一个连续的值**。

![https://img-blog.csdn.net/20180914152158467?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

下面是另一种监督学习的例子：

假设你想看医疗记录，并且设法预测乳腺癌是恶性的还是良性的，假设某人发现了一个乳腺肿瘤，即乳房上的肿块，恶性肿瘤就是有害的且危险的，良性肿瘤就是无害的。横轴是肿瘤的尺寸，纵轴的1代表恶性，0代表良性。机器学习的问题就是让你根据粉色点所处的肿瘤尺寸，计算肿瘤是良性的还是恶性的概率。用更专业的术语来说，这就是一个分类问题，分类是指**设法预测一个离散值**。这个例子中，只有一个特征或者说是只有一个属性，即通过肿瘤的大小这一个属性来预测肿瘤是恶性的还是良性的。

![https://img-blog.csdn.net/20180914164410611?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

在其他的机器学习问题中，可能会有多个特征，多个属性，例如现在假设我们不仅知道肿瘤的大小，还知道病人的年纪。以O代表良性肿瘤，用X代表恶性肿瘤。横轴代表肿瘤的大小，纵轴代表病人的年纪。

![https://img-blog.csdn.net/20180914164732690?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

对于这种有多个特征的数据集，学习算法能够做的就是在数据上画出一条直线，设法将恶性肿瘤和良性肿瘤分开。这样对于紫色的点，通过这种方式，你的学习算法就会认为，这个肿瘤位于良性区域，因此这个肿瘤是良性的机率比恶性的大。

**在分类算法中目标变量的类型通常是标称型(如：真与假)，而在回归算法中通常是连续型(如：1~100)。**

#### Unsupervised Learning：

对于监督学习中的每一个样本，我们已经被清楚地告知了什么是所谓的正确答案，即它们是良性还是恶性，而在无监督学习中，我们所用的数据没有任何的标签，或者是都具有相同的标签或者都没有标签，我们拿到这个数据集，但我们却不知道要拿它来做什么，也不知道每个数据点究竟是什么，我们只是知道有一个这样的数据集，例如下面这幅图，对于给定的数据集，无监督学习算法可能判定该数据集包含两个不同的簇，无监督学习算法可以把这些数据分成两个不同的簇，这就是聚类算法。聚类算法被应用在很多地方，其中一个应用聚类算法的例子就是谷歌新闻，谷歌新闻所做的就是每天去网络上，收集几万条甚至几十万条新闻，然后将它们组合成一个个新闻专题，谷歌新闻所做的就是去搜索成千上万条新闻，然后自动将它们分簇，有关同一主题的新闻被显示在一起。

![https://img-blog.csdn.net/20180914214933987?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

其实聚类算法和无监督学习算法也可应用到许多其他的问题，例如它在基因组学中的应用，下图是一个DNA微阵列数据的例子，给定一组不同的个体，对于每个个体，检测他们是否拥有某个特定的基因，也就是检测特定基因的表达程度，这些红，灰，绿等颜色展示了不同个体拥有特定基因的程度，然后你需要做的就是运行一个聚类算法，把不同的个体归入不同的类也就是归为不同类型的人，这就是无监督学习。因为我们只知道这里有一堆数据，但是不知道这些数据是什么，不知道每一个是哪一个类型，甚至不知道有哪些类型。因为我们没有把数据集的所谓的“正确答案”给算法，所以这就是无监督学习。

![https://img-blog.csdn.net/20180914215834553?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

## 2.   Linear regression with one  variable（单变量线性回归）：

### 2.1 Model   representaion（模型表达）：

让我们先来看之前那个预测住房价格的例子，假设你一个朋友有一套大小为1250平方英尺大小的房子，他想要让你帮他预测这个房子能卖多少钱？你可以用一条直线来拟合这组数据，根据这个模型，你可以告诉你的朋友，他的房子也许可以卖到220000美元左右。这是一个监督学习算法的例子，它之所以是监督学习，是因为每一个例子都有一个“正确的答案”，也就是说我们知道数据集中所卖的房子的实际大小和价格，而且这也是一个回归问题的例子，回归是指预测一个具体的数值输出，也就是房子的价格。另一种最常见的监督学习问题被称为分类问题，用来预测离散值的输出，例如我们观察肿瘤，并试图判断它是良性的还是恶性的。

![https://img-blog.csdn.net/20180915115357701?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

更正式一点说，在监督学习里，我们有一个数据集，它被称为一个训练集（Training Set），以住房价格为例，我们有一个房价的训练集，我们的工作是从这个数据中，学习如何预测房价。

下面将定义一些符号将更有利于理解：

![https://img-blog.csdn.net/20180915162410546?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

**m：表示训练样本的数量，也就是训练集中的实例数量。**

**x：代表输入变量或者说是特征。**

**y：代表输出变量也就是要预测的目标变量。**

**(x，y）：表示一个训练样本。**

**(x(i),y(i) )：代表第i个训练样本。（这个i是训练集中的一个索引，指的是表格中的第i行）。**

## 监督学习算法的工作流程：

向学习算法提供训练集，比如说房价训练集，学习算法的任务是输出一个函数，通常用小写字母h表示，h代表假设函数，在房价预测的例子中假设函数的作用是，把房子的大小作为输入变量，将它作为x的值，然后输出这个房子预测的y值，h是一个从x映射到y的函数。在这个例子中，假设函数是预测y是关于x的线性函数。**hθ(x)也可以简写为h(x)。**

![https://img-blog.csdn.net/20180915163516220?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

### 2.2 Cost  funcion（代价函数）：

代价函数是为了帮助我们弄清楚如何把最有可能的直线与我们的数据相拟合。

例如我们有一个这样的训练集，m代表了训练样本的数量，假设m=47，我们的假设函数是![https://img-blog.csdn.net/20180915164550750?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]
这些θ0和θ1我们把它们称为模型参数，我们需要做的就是通过选择合适的θ0和θ1，使得误差尽量的小。选择不同的θ0和θ1，会得到不同的假设函数，例如可以选择下图中的这些θ0和θ1。

![https://img-blog.csdn.net/20180915165313682?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

在线性回归中，我们有一个训练集，我们要做的就是得出θ0和θ1这两个参数的值，使得假设函数所表示的直线尽量地与这些数据点能够很好的拟合，例如就像下图中的这一条直线，那么如何求出θ0和θ1的值来使它很好地拟合数据呢？

![https://img-blog.csdn.net/20180915172348636?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

我们的想法是，我们需要选择能使h(x)，也就是输入x时，我们预测的值最接近该样本对应的y值的参数θ0和θ1。所以，在我们的训练集中我们会得到一定数量的样本，我们知道x表示卖出哪所房子，并且知道卖出的这所房子的实际交易价格。所以我们要尽量选择合适的参数值，使得在训练集中，对于给出的训练集中的x值，我们能够合理准确地预测y的值。接下来给出标准的定义，在线性回归中，我们要解决的实际上是一个最小化的问题，我们要做的就是尽量减少假设输出与房子的真实交易价格之间的差的平方。所以我们要做的就是对所有的训练样本进行一个求和，将第i号对应的预测结果减去第i号房子的实际交易价格所得的差的平方相加得到一个总和，而我们希望这个总和能够尽可能的小，也就是预测值和实际值的差的平方误差和或者说预测价格和实际卖出价格的差的平方。为了让表达式的数学意义变得容易理解一点，实际上考虑的是所求误差和的1/m，也就是我们要尝试尽量减少平均误差，也就是尽量减少其1/(2*m)，即通常是平均误差的一半，这只是为了使数学意义更直白一些，因此对这个求和值的1/2求最小值，也能够得到相同的θ0和θ1。所以代价函数可以表示为
![https://img-blog.csdn.net/20180915171629331?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

所以我们要做的就是对J(θ0,θ1)求最小值，这就是代价函数。代价函数也被称作平方误差函数，有时也被称为平方误差代价函数，事实上，之所以要求出误差的平方和，是因为误差平方代价函数对于大多数问题，特别是回归问题，都是一个合理的选择，当然还有其他的代价函数也能很好地发挥作用，但是平方误差代价函数可能是解决回归问题最常用的手段了。

为了更好地使代价函数J可视化，使用一个简化的假设函数h，设θ0=0，则h(x) = θ1 * x。使用简化的代价函数便于更好的理解代价函数的概念。

![https://img-blog.csdn.net/20180916090156208?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

下图(左)中的红色X代表房子实际交易价格，θ0=0时，假设函数是过原点的直线，对于每一条假设函数，它的预测值和实际值的差的平方也就是下图(左)中蓝色竖线的长度的平方。在下图(右)做出相应的代价函数J(θ1)的图像，显然，当θ1=1时，J(θ1)最小。

![https://img-blog.csdn.net/20180916091429981?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

![https://img-blog.csdn.net/20180916093533292?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

学习算法的优化目的是找到一个最优的θ1，使得J(θ1)最小化，显然由图可知当θ1=1时，J(θ1)取得最小值。 我们绘制一个等高线图，三个坐标分别为θ0和θ1和J(θ0,θ1)。如下图所示，这是一个类似碗状形状的图像，竖直高度即是J(θ0,θ1)。

![https://img-blog.csdn.net/20180916093959310?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

也可以在二维图中用下图进行表示，每一个椭圆形显示了一系列J(θ0,θ1)值相等的点，例如下图右中的三个点，他们的J(θ0,θ1)值相等，这些同心椭圆的中心点(碗状底部)就是最小值。

![https://img-blog.csdn.net/20180916094718119?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

![https://img-blog.csdn.net/20180916095334776?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

![https://img-blog.csdn.net/2018091609543185?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

![https://img-blog.csdn.net/20180916095708979?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

### 2.3   Gradient   descent（梯度下降法）：

梯度下降法可以最小化代价函数J。为了简洁起见，假设这里只有两个参数θ0和θ1，梯度下降的思路是先给θ0和θ1赋初值，到底需要赋什么值其实并不重要，但通常的选择是让θ0设为0，θ1也设为0，然后不停地一点点地改变θ0和θ1来使得J(θ0,θ1)变小，直到我们找到J(θ0,θ1)的最小值或者局部最小值。

![https://img-blog.csdn.net/2018091610201493?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

让我们通过一些图片来看看梯度下降法是如何工作的，假设要让下图这个函数值最小化。

![https://img-blog.csdn.net/20180916102438470?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

首先对θ0和θ1赋以某个初值，也就是对应于从这个函数表面上的某个点出发，不管θ0和θ1的取值是多少，假设将它们都初始化为0，但有时你也可以把它们初始化为其他值。现在把这个图像想象为一座山，想象一下你正站在下图所示点处。在梯度下降算法中，我们要做的就是旋转360度，看看周围，并问自己，如果在某个方向上走一小步，朝哪个方向走才能尽快走下山？

![https://img-blog.csdn.net/20180916102742617?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

每次都往倾斜度最大的方向迈一小步，直到到达局部最低点，所以所走路径大致为下图。

![https://img-blog.csdn.net/20180916103421912?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

梯度下降有一个有趣的特点，对于不同的起始点，可能会得到不同的局部最优解，例如下图。

![https://img-blog.csdn.net/20180916103704882?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

![https://img-blog.csdn.net/20180916105736346?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

上图就是梯度下降算法的数学定义，这个公式有很多细节问题，首先注意这个符号“:=”，我们使用“:=”表示赋值，这是一个赋值运算符，例如a:=b，在计算机中，这意味着将b的值赋给a。而如果写a=b，那么这是一个真假判定，其实就是在断言a的值等于b的值。这里的α是学习率（learning rate），α用来控制梯度下降时，我们需要迈出多大的步子。如果α值很大，梯度下降就很迅速，也就是我们会用大步子下山。如果α值很小，我们会用小碎步下山。还有一个关于梯度下降法的细节，我们要更新θ0和θ1，当j=0和j=1时，会产生更新，实现梯度下降算法的微妙之处是，对于这个表达式，我们需要同时更新θ0和θ1，θ0更新为θ0减去某项，θ1更新为θ1减去某项，实现方法是计算表达式等式右边的部分，然后同时更新θ0和θ1。事实发现，同步更新是更自然的实现方法。如果用非同步更新去实现算法，可能也会正确工作，但是这种方法并不是人们通常用的那个梯度下降算法，而是具有不同性质的其他算法。由于各种原因，这其中会表现出微小的差别。所以我们需要做的就是在梯度下降算法中实现同步更新。

接下来进一步解释这个数学定义：
![https://img-blog.csdn.net/20180916144452222?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

这是关于实数θ1的代价函数J(θ1)，横轴表示θ1，纵轴表示J(θ1)。第一幅图中从θ1处开始梯度下降，梯度下降要做的就是不断更新θ1。在微积分中这一点的导数其实就是这一点的切线，就是那条红色的直线，这条直线的斜率为正，也就是说它有正导数，所以θ1就更新为θ1-α*(一个正数)，α也就是学习速率永远是一个正数，所以我们就相当于将θ1左移，使θ1变小了。我们看到这么做是对的，因为往左移实际上是在向最低点靠近。在第二幅图中，用同样的代价函数J(θ1)，再作出同样的图像。而这次把参数初始化到左边那个点，现在的导数项也就是这点的切线的斜率，这条切线的斜率是负的，所以这个函数有负导数，这个导数项是小于0的，所以当更新θ1时，θ1被更新为θ1-α*(一个负数)，所以实际上θ1在增大，也就是向右移动，也是在接近最低点。这就是导数项的意义。

接下来看一看学习速率α的作用：

![https://img-blog.csdn.net/20180916150733683?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

如果α太小，就会用一个比较小的系数来更新，梯度下降可能就会很慢，就需要很多步才能到达全局最低点。

如果α太大，如图所示，那么梯度下降可能会越过最低点，甚至可能无法收敛或者发散。

![https://img-blog.csdn.net/20180916151257490?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

如果θ1已经处在一个局部最优点，局部最优点的导数为0，而在梯度下降更新过程中，θ1更新为θ1-α*0，所以θ1将不会改变。即如果参数已经处于局部最低点，那么梯度下降法更新其实什么都没做，它不会改变参数的值，它使解始终保持在局部最优点。这也解释了，即使学习速率α保持不变，梯度下降法也可以收敛到局部最低点的原因。

我们来看一个例子：

![https://img-blog.csdn.net/20180916152253830?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

首先在品红色的那个点初始化θ1的值，然后用梯度下降法一步步更新，你会发现，越接近最低点，斜率越小，导数也就越小，逐渐接近于0。所以随着梯度下降法的进行，移动的幅度会自动变得越来越小，直到最终移动幅度非常小，你会发现，已经收敛到了局部最小值，这就是梯度下降的运行方式。所以实际上没有必要再另外减小α，这就是梯度下降算法。你可以用它来尝试最小化任何代价函数J，而不只是线性回归中的代价函数J。

## Gradient descent for linear regression:

现在我们将梯度下降和线性回归结合，得到线性回归的算法，它可以用直线模型来拟合数据。下图是梯度下降法和线性回归模型（包括线性假设和平方差代价函数），我们要做的就是将梯度下降法应用到最小化平方差代价函数。

![https://img-blog.csdn.net/20180917085852219?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

为了应用梯度下降法，关键步骤就是求出左边式子中的导数项。因此我们需要知道这个偏导数项是什么，J(θ0,θ1)对θ0和θ1的偏导数如下：

![https://img-blog.csdn.net/20180917090405360?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

那么梯度下降算法就可以写成下面这种形式，这就是线性回归算法：

![https://img-blog.csdn.net/20180917090609310?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

这个算法也叫批量梯度下降法(batch gradient descent)，就是不断重复这个步骤，直到得到最优值。

![https://img-blog.csdn.net/20180917091233142?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]

执行梯度下降时，根据你的初始值的不同，可能会得到不同的局部最优解。

但线性回归的代价函数总是一个弓状函数，术语叫做凸函数。这个函数没有局部最优解，只有一个全局最优解。当你计算这种代价函数的梯度下降的时候，只要你使用线性回归，它总会收敛到全局最优，因为没有全局最优解之外的局部最优解。

![https://img-blog.csdn.net/20180917091716299?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzExNjA0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70]




