torch.backends.cudnn.deterministic是啥？顾名思义，将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。

* torch.backends.cudnn.benchmark=True

设置 `torch.backends.cudnn.benchmark=True` 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。

#### DROPOUT

> *CLASS*`torch.nn.``Dropout`(*p=0.5*, *inplace=False*)
>
> During training, randomly zeroes some of the elements of the input tensor with probability `p` using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.
>
> 用来正则化防止过拟合。

#### model.eval()

> 保证在做batch normalization 时用的是所有数据的均值和方差

#### model.train()

> 如果模型中有BN层(Batch Normalization）和 Dropout，需要在训练时添加`model.train()`。`model.train()`是保证BN层能够用到每一批数据的均值和方差。对于Dropout，`model.train()`是随机取一部分网络连接来训练更新参数。

#### 随机数种子

> random随机数是这样生成的：我们将这套复杂的算法（是叫随机数生成器吧）看成一个黑盒，把我们准备好的种子扔进去，它会返给你两个东西，一个是你想要的随机数，另一个是保证能生成下一个随机数的新的种子，把新的种子放进黑盒，又得到一个新的随机数和一个新的种子，从此在生成随机数的路上越走越远。
> 原文链接：https://blog.csdn.net/qq_41375609/article/details/99327074
>
> 深度学习网络模型中**初始的权值参数**通常都是**初始化成随机数**而使用梯度下降法最终得到的局部最优解**对于初始位置点的选择很敏感**为了能够完全复现作者的开源深度学习代码，随机种子的选择能够减少一定程度上算法结果的随机性，也就是更接近于原始作者的结果
>
> ```
>     torch.manual_seed(seed) # 设置 cpu 的随机数种子
>     torch.cuda.manual_seed(seed) # 对于单张显卡，设置 gpu 的随机数种子
>     torch.cuda.manual_seed_all(seed) # 对于多张显卡，设置所有 gpu 的随机数种子
> ```