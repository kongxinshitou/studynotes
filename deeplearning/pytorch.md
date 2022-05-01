

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

##### Pillow读图片

首先，我们需要使用 Pillow 中的下述代码读入上面的图片。

```python

from PIL import Image
im = Image.open('jk.jpg')
im.size
输出: 318, 116
```

Pillow 是以二进制形式读入保存的，那怎么转为 NumPy 格式呢？这个并不难，我们只需要利用 NumPy 的 asarray 方法，就可以将 Pillow 的数据转换为 NumPy 的数组格式

```python

import numpy as np

im_pillow = np.asarray(im)

im_pillow.shape
输出：(116, 318, 3)
```

Pillow 读入后通道的顺序就是 R、G、B，而 OpenCV 读入后顺序是 B、G、R。

## pytorch基础

#####  创建tensor

```python
torch.tensor(data,dtype=None,device=None,requires_grad=False)
```

#####从numpy中创建tensor

```python
torch.from_numpy(ndarray)
```

#####创建零矩阵tensor

```python
torch.zeros(*size,dtype=None...)
```

#####创建单位矩阵

```python
torch.eye(size,dtype=None...)
```

#####创建全一矩阵Tensor

```python
torch.ones(size,dtype=None...)
```

#####创建随机矩阵

```python
torch.rand(size)
#随机生成的浮点数据是0~1区间均匀分布
torch.randn(size)
#随机生成的浮点数据是均值为0方差为1的标准正态分布
torch.normal(mean,std,size)
#可以指定均值和标准差的正态分布
torch.randint(low,high,size)
#随机生成的整数是[low,high)均匀生成的随机整数
```

##### int和tensor的转换

```python
a = torch.tensor(1)
a = a.item()
#torch.tentor把一个数字转换成一个tensor,tensor.item()把tensor转换回数字
```

##### list和tensor的转换

```python
a = [1,2,3]
b = torch.tensor(a)
c = b.numpy().tolist()
#tensor.numpy()把tensor转换成numpy()结构，ndarray用tolist()换为列表
```



##### 矩阵转置

* transpose()

```python
x = torch.rand(2,3,5)
x.shape
x = x.permute(2,1,0)
x.shape
#2 表示原来第二个维度现在放在了第零个维度；同理 1 表示原来第一个维度仍旧在第一个维度；0 表示原来第 0 个维度放在了现在的第 2 个维度，形状就变成了[5,3,2]
```

* transpose

```python
x = x.transpose(1,0)
#交换维度1和维度0的数据
```

* 注意，经过transpose和permute之后的数据变得不再连续

```python
x = torch.rand(8,2)
x = x.permute(1,0)

```

##### 查看torch版本，查看cuda是否可用

```python
print(torch.__version__)
print(torch.cuda.is_available)
```

##### squeeze()

```python
a = torch.rand(2,1,3)
a
tensor([[[0.7861, 0.8679, 0.7388]],

        [[0.1943, 0.3558, 0.1251]]])
a = a.squeeze(1)
a
tensor([[0.7861, 0.8679, 0.7388],
       [0.1943, 0.3558, 0.1251]])
```

将第一个维度的的数据删除成功因为第1维度的大小是1

##### unsqueeze()

```python
>>> x = torch.rand(2,1,3)
>>> y = x.unsqueeze(2)
>>> y.shapetorch.Size([2, 1, 1, 3])
```

给指定位置加上维度为1的数据，原本的维度向后移

#### Tensor的连接操作

##### cat

```python
torch.cat(tensors,dim=0,out=None)
```

* 两个重要参数
  * tensor,若干个准备拼接的tensor
  * dim,维度

```python

>>> A=torch.ones(3,3)
>>> B=2*torch.ones(3,3)
>>> A
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
>>> B
tensor([[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]])

>>> C=torch.cat((A,B),0)
>>> C
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]])

```

按维度0进行复制

##### stack

* cat是将多个tensor在已有的维度上进行连接，增加维度要用到stack

```python
torch.stack(inputs,dim=0)
```

* inputs表示要拼接的tensor，dim表示新建立维度的方向

```python

>>> A=torch.arange(0,4)
>>> A
tensor([0, 1, 2, 3])
>>> B=torch.arange(5,9)
>>> B
tensor([5, 6, 7, 8])
>>> C=torch.stack((A,B),0)
>>> C
tensor([[0, 1, 2, 3],
        [5, 6, 7, 8]])
>>> D=torch.stack((A,B),1)
>>> D
tensor([[0, 5],
        [1, 6],
        [2, 7],
        [3, 8]])
```

#### Tensor的切分操作

##### chunk

* chunk的作用就是将tensor按照声明的dim进行尽可能的平均的划分

```python
torch.chunk(input,chunks,dim=0)
```

* input表示要做chunk操作的tensor
* chunks代表将要被划分的块的数量

```python

>>> A=torch.tensor([1,2,3,4,5,6,7,8,9,10])
>>> B = torch.chunk(A, 2, 0)
>>> B
(tensor([1, 2, 3, 4, 5]), tensor([ 6,  7,  8,  9, 10]))
```

长度为10的划分为两个长度为2的

```python

>>> B = torch.chunk(A, 3, 0)
>>> B
(tensor([1, 2, 3, 4]), tensor([5, 6, 7, 8]), tensor([ 9, 10]))
```

先做除法然后向上取整，最后的不够的放在一块

```python

>>> A=torch.tensor([1,2,3])
>>> B = torch.chunk(A, 5, 0)
>>> B
(tensor([1]), tensor([2]), tensor([3]))
```

chunk的参数大于Tensor可以切分的长度，则被切分为若干个为1的向量

```python
>>>A = torch.rand(4,4)
>>>A
tensor([[0.2299, 0.6047, 0.9569, 0.5646],
        [0.1229, 0.0749, 0.3185, 0.6662],
        [0.3831, 0.2937, 0.5499, 0.6194],
        [0.5323, 0.1381, 0.5562, 0.0943]])
>>>B = torch.chunk(A,2,0)
>>>B
(tensor([[0.2299, 0.6047, 0.9569, 0.5646],
         [0.1229, 0.0749, 0.3185, 0.6662]]),
 tensor([[0.3831, 0.2937, 0.5499, 0.6194],
         [0.5323, 0.1381, 0.5562, 0.0943]]))
```

表示第dim维度上进行切分

##### split

```python

torch.split(tensor, split_size_or_sections, dim=0)
```

* 首先是 tensor，也就是待切分的 Tensor。
* 然后是 split_size_or_sections 这个参数。当它为整数时，表示将 tensor 按照每块大小为这个整数的数值来切割；当这个参数为列表时，则表示将此 tensor 切成和列表中元素一样大小的块。
* 最后同样是 dim，它定义了要按哪个维度切分。

```python

>>> C=torch.split(A, 3, 0)
>>> C
(tensor([[0.6418, 0.4171, 0.7372, 0.0733],
        [0.0935, 0.2372, 0.6912, 0.8677],
        [0.5263, 0.4145, 0.9292, 0.5671]]), 
tensor([[0.2284, 0.6938, 0.0956, 0.3823]]))
```

PyTorch 会尽可能凑够每一个结果，使得其对应 dim 的数据大小等于 split_size_or_sections

```python

>>> A=torch.rand(5,4)
>>> A
tensor([[0.1005, 0.9666, 0.5322, 0.6775],
        [0.4990, 0.8725, 0.5627, 0.8360],
        [0.3427, 0.9351, 0.7291, 0.7306],
        [0.7939, 0.3007, 0.7258, 0.9482],
        [0.7249, 0.7534, 0.0027, 0.7793]])
>>> B=torch.split(A,(2,3),0)
>>> B
(tensor([[0.1005, 0.9666, 0.5322, 0.6775],
        [0.4990, 0.8725, 0.5627, 0.8360]]), 
tensor([[0.3427, 0.9351, 0.7291, 0.7306],
        [0.7939, 0.3007, 0.7258, 0.9482],
        [0.7249, 0.7534, 0.0027, 0.7793]]))
```

就是将 Tensor A，沿着第 0 维进行切分，每一个结果对应维度上的尺寸或者说大小，分别是 2（行），3（行）。

##### unbind

```python

torch.unbind(input, dim=0)
```

* 其中，input 表示待处理的 Tensor，dim 还是跟前面的函数一样，表示切片的方向

```python

>>> A=torch.arange(0,16).view(4,4)
>>> A
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])
>>> b=torch.unbind(A, 0)
>>> b
(tensor([0, 1, 2, 3]), 
tensor([4, 5, 6, 7]), 
tensor([ 8,  9, 10, 11]), 
tensor([12, 13, 14, 15]))
```

在这个例子中，我们首先创建了一个 4x4 的二维矩阵 Tensor，随后我们从第 0 维，也就是“行”的方向进行切分 ，因为矩阵有 4 行，所以就会得到 4 个结果。

```python

>>> b=torch.unbind(A, 1)
>>> b
(tensor([ 0,  4,  8, 12]), 
tensor([ 1,  5,  9, 13]), 
tensor([ 2,  6, 10, 14]), 
tensor([ 3,  7, 11, 15]))
```

不难发现，这里是按照“列”的方向进行拆解的。所以，unbind 是一种降维切分的方式，相当于删除一个维度之后的结果

#### Tensor的索引操作

##### index_select

```python

torch.index_select(tensor, dim, index)
```

这里的 tensor、dim 跟前面函数里的一样，不再赘述。我们重点看一看 index，它表示从 dim 维度中的哪些位置选择数据，这里需要注意，index是 torch.Tensor 类型

```python

>>> A=torch.arange(0,16).view(4,4)
>>> A
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])
>>> B=torch.index_select(A,0,torch.tensor([1,3]))
>>> B
tensor([[ 4,  5,  6,  7],
        [12, 13, 14, 15]])
>>> C=torch.index_select(A,1,torch.tensor([0,3]))
>>> C
tensor([[ 0,  3],
        [ 4,  7],
        [ 8, 11],
        [12, 15]])
```

在这个例子中，我们先创建了一个 4x4 大小的矩阵 Tensor A。然后，我们从第 0 维选择第 1（行）和 3（行）的数据，并得到了最终的 Tensor B，其大小为 2x4。随后我们从 Tensor A 中选择第 0（列）和 3（列）的数据，得到了最终的 Tensor C，其大小为 4x2

##### masked_select

```python
 torch.masked_select(input, mask, out=None)

```

input 表示待处理的 Tensor。mask 代表掩码张量，也就是满足条件的特征掩码。这里你需要注意的是，mask 须跟 input 张量有相同数量的元素数目，但形状或维度不需要相同

```python

>>> A=torch.rand(5)
>>> A
tensor([0.3731, 0.4826, 0.3579, 0.4215, 0.2285])
>>> B=A>0.3
>>> B
tensor([ True,  True,  True,  True, False])
```

在这段代码里，我们让 A 跟 0.3 做比较，得到了一个新的 Tensor，内部每一个数值表示的是 A 中对应数值是否大于 0.3。

```python

>>> C=torch.masked_select(A, B)
>>> C
tensor([0.3731, 0.4826, 0.3579, 0.4215])
```

你会发现，C 实际上得到的就是：A 中“满足 B 里面元素值为 True 的”对应位置的数据。

```python

>>> A=torch.rand(5)
>>> A
tensor([0.3731, 0.4826, 0.3579, 0.4215, 0.2285])
>>> C=torch.masked_select(A, A>0.3)
>>> C
tensor([0.3731, 0.4826, 0.3579, 0.4215])
```

#### 数据类型转换

#####PIL.Image和Numpy.ndarray 转换

那么如何将 PIL.Image 或 Numpy.ndarray 格式的数据转化为 Tensor 格式呢？

这需要用到transforms.ToTensor() 类。而反之，将 Tensor 或 Numpy.ndarray 格式的数据转化为 PIL.Image 格式，则使用transforms.ToPILImage(mode=None) 类。

它则是 ToTensor 的一个逆操作，它能把 Tensor 或 Numpy 的数组转换成 PIL.Image 对象。其中，参数 mode 代表 PIL.Image 的模式，如果 mode 为 None（默认值），则根据输入数据的维度进行推断：

* 输入为 3 通道：mode 为’RGB’；
* 输入为 4 通道：mode 为’RGBA’；
* 输入为 2 通道：mode 为’LA’;
* 输入为单通道：mode 根据输入数据的类型确定具体模式。

```python

from PIL import Image
from torchvision import transforms 



img = Image.open('jk.jpg') 
display(img)
print(type(img)) # PIL.Image.Image是PIL.JpegImagePlugin.JpegImageFile的基类
'''
输出: 
<class 'PIL.JpegImagePlugin.JpegImageFile'>
'''

# PIL.Image转换为Tensor
img1 = transforms.ToTensor()(img)
print(type(img1))
'''
输出: 
<class 'torch.Tensor'>
'''

# Tensor转换为PIL.Image
img2 = transforms.ToPILImage()(img1)  #PIL.Image.Image
print(type(img2))
'''
输出: 
<class 'PIL.Image.Image'>
'''
```

##### Resize

```python

torchvision.transforms.Resize(size, interpolation=2)
```

* size：期望输出的尺寸。如果 size 是一个像 (h, w) 这样的元组，则图像输出尺寸将与之匹配。如果 size 是一个 int 类型的整数，图像较小的边将被匹配到该整数，另一条边按比例缩放。
* interpolation：插值算法，int 类型，默认为 2，表示 PIL.Image.BILINEAR。

```python

from PIL import Image
from torchvision import transforms 

# 定义一个Resize操作
resize_img_oper = transforms.Resize((200,200), interpolation=2)

# 原图
orig_img = Image.open('jk.jpg') 
display(orig_img)

# Resize操作后的图
img = resize_img_oper(orig_img)
display(img)
```

##### 剪裁

torchvision.transforms提供了多种剪裁方法，例如中心剪裁、随机剪裁、四角和中心剪裁等。我们依次来看下它们的定义。

* 中心剪裁

```python

torchvision.transforms.CenterCrop(size)
```

其中，size 表示期望输出的剪裁尺寸。如果 size 是一个像 (h, w) 这样的元组，则剪裁后的图像尺寸将与之匹配。如果  size  是  int  类型的整数，剪裁出来的图像是  (size, size)  的正方形。

* 随机剪裁

```python

torchvision.transforms.RandomCrop(size, padding=None)
```

其中，size 代表期望输出的剪裁尺寸，用法同上。而 padding 表示图像的每个边框上的可选填充。默认值是 None，即没有填充。通常来说，不会用 padding 这个参数，至少对于我来说至今没用过

* 五块剪裁

```python

torchvision.transforms.FiveCrop(size)
```

最后要说的是 FiveCrop，我们将给定的 PIL Image 或 Tensor ，分别从四角和中心进行剪裁，共剪裁成五块，定义如下：

```python

from PIL import Image
from torchvision import transforms 

# 定义剪裁操作
center_crop_oper = transforms.CenterCrop((60,70))
random_crop_oper = transforms.RandomCrop((80,80))
five_crop_oper = transforms.FiveCrop((60,70))

# 原图
orig_img = Image.open('jk.jpg') 
display(orig_img)

# 中心剪裁
img1 = center_crop_oper(orig_img)
display(img1)
# 随机剪裁
img2 = random_crop_oper(orig_img)
display(img2)
# 四角和中心剪裁
imgs = five_crop_oper(orig_img)
for img in imgs:
    display(img)
```

##### 翻转

* 以概率 p 随机水平翻转图像，定义如下：

```python

torchvision.transforms.RandomHorizontalFlip(p=0.5)
```

* 以概率 p 随机垂直翻转图像，定义如下：

```python

torchvision.transforms.RandomVerticalFlip(p=0.5)
```

```python

from PIL import Image
from torchvision import transforms 

# 定义翻转操作
h_flip_oper = transforms.RandomHorizontalFlip(p=1)
v_flip_oper = transforms.RandomVerticalFlip(p=1)

# 原图
orig_img = Image.open('jk.jpg') 
display(orig_img)

# 水平翻转
img1 = h_flip_oper(orig_img)
display(img1)
# 垂直翻转
img2 = v_flip_oper(orig_img)
display(img2)
```

#### 只对Tensor进行变换

##### 标准化

output=(input−mean)/std

```python

torchvision.transforms.Normalize(mean, std, inplace=False)
```

其中，每个参数的含义如下所示：

* mean：表示各通道的均值；
* std：表示各通道的标准差；
* inplace：表示是否原地操作，默认为否。

```python

from PIL import Image
from torchvision import transforms 

# 定义标准化操作
norm_oper = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# 原图
orig_img = Image.open('jk.jpg') 
display(orig_img)

# 图像转化为Tensor
img_tensor = transforms.ToTensor()(orig_img)

# 标准化
tensor_norm = norm_oper(img_tensor)

# Tensor转化为图像
img_norm = transforms.ToPILImage()(tensor_norm)
display(img_norm)
```

注意：标准化要Tensor

#### 变换的组合

Compose 类是将多个变换组合到一起，它的定义如下。

```python

torchvision.transforms.Compose(transforms)
```

其中，transforms 是一个 Transform 对象的列表，表示要组合的变换列表。我们还是结合例子动手试试，如果我们想要将图片变为 200*200 像素大小，并且随机裁切成 80 像素的正方形。那么我们可以组合 Resize 和 RandomCrop 变换，具体代码如下所示

```python

from PIL import Image
from torchvision import transforms 

# 原图
orig_img = Image.open('jk.jpg') 
display(orig_img)

# 定义组合操作
composed = transforms.Compose([transforms.Resize((200, 200)),
                               transforms.RandomCrop(80)])

# 组合操作后的图
img = composed(orig_img)
display(img)
```

##### 结合datasets使用

在利用torchvision.datasets 读取 MNIST 数据集时，有一个参数“transform”吗？它就是用于对图像进行预处理操作的，例如数据增强、归一化、旋转或缩放等。这里的“transform”就可以接收一个torchvision.transforms操作或者由 Compose 类所定义的操作组合。

```python

from torchvision import transforms
from torchvision import datasets

# 定义一个transform
my_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5), (0.5))
                                  ])
# 读取MNIST数据集 同时做数据变换
mnist_dataset = datasets.MNIST(root='./data',
                               train=False,
                               transform=my_transform,
                               target_transform=None,
                               download=True)

# 查看变换后的数据类型
item = mnist_dataset.__getitem__(0)
print(type(item[0]))
'''
输出：
<class 'torch.Tensor'>
'''
```

##### pytorch中的卷积

让我们先看看创建一个 nn.Conv2d 需要哪些必须的参数：

```python

# Conv2d类
class torch.nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size, 
                      stride=1, 
                      padding=0, 
                      dilation=1, 
                      groups=1, 
                      bias=True, 
                      padding_mode='zeros', 
                      device=None, 
                      dtype=None)

```

我们挨个说说这些参数。首先是跟通道相关的两个参数：in_channels 是指输入特征图的通道数，数据类型为 int，在标准卷积的讲解中 in_channels 为 m；

out_channels 是输出特征图的通道数，数据类型为 int，在标准卷积的讲解中 out_channels 为 n。

kernel_size 是卷积核的大小，数据类型为 int 或 tuple，需要注意的是只给定卷积核的高与宽即可，在标准卷积的讲解中 kernel_size 为 k。

stride 为滑动的步长，数据类型为 int 或 tuple，默认是 1，在前面的例子中步长都为 1。

padding 为补零的方式，注意当 padding 为’valid’或’same’时，stride 必须为 1。

对于 kernel_size、stride、padding 都可以是 tuple 类型，当为 tuple 类型时，第一个维度用于 height 的信息，第二个维度时用于 width 的信息。bias 是否使用偏移项

##### 构建神经网络的知识点

1. 必须继承nn.Module类
2. 重写\__init\__()方法。把要学习的参数的层放到构造函数中
3. forward()是必须重写的方法，对于不需要学习参数的层，一般来说放在这里，BN层，激活函数和Dropout。

##### model.named_parameters()

* 存放模型可训练的参数

##### 保存模型的参数

```python

torch.save(model.state_dict(), './linear_model.pth')
```

第一个参数是模型的 state_dict，而第二个参数要保存的位置。代码中的 state_dict 是一个字典，在模型被定义之后会自动生成，存储的是模型可训练的参数。

加载模型的方式如下所示:

```python

# 先定义网络结构
linear_model = LinearModel()
# 加载保存的参数
linear_model.load_state_dict(torch.load('./linear_model.pth'))
linear_model.eval()
for parameter in linear_model.named_parameters():
  print(parameter)
输出：
('weight', Parameter containing:
tensor([[2.0071]], requires_grad=True))
('bias', Parameter containing:
tensor([3.1690], requires_grad=True))
```

#### 为什么要model.eval()

这里有个 model.eval() 需要你注意一下，因为有些层（例如，Dropout 与 BN）在训练时与评估时的状态是不一样的，当进入评估时要执行 model.eval()，模型才能进入评估状态。这里说的评估不光光指代评估模型，也包括模型上线时候时的状态。

##### 保存网络结构和参数

相比第一种方式，这种方式在加载模型的时候，不需要加载网络结构了。具体代码如下所示：

```python

# 保存整个模型
torch.save(model, './linear_model_with_arc.pth')
# 加载模型，不需要创建网络了
linear_model_2 = torch.load('./linear_model_with_arc.pth')
linear_model_2.eval()
for parameter in linear_model_2.named_parameters():
  print(parameter)
# 输出：
('weight', Parameter containing:
tensor([[2.0071]], requires_grad=True))
('bias', Parameter containing:
tensor([3.1690], requires_grad=True))
```

##### 微调

* 预训练后的模型只需要很少的训练集就可以得到很好的效果，微调最关键的一步是调整最后全连接层输出的错误。

#### 计算梯度注意点

- We can only obtain the `grad` properties for the leaf nodes of the computational graph, which have `requires_grad` property set to `True`. For all other nodes in our graph, gradients will not be available.
- We can only perform gradient calculations using `backward` once on a given graph, for performance reasons. If we need to do several `backward` calls on the same graph, we need to pass `retain_graph=True` to the `backward` call

##### torch.no_grad()

By default, all tensors with `requires_grad=True` are tracking their computational history and support gradient computation. However, there are some cases when we do not need to do that, for example, when we have trained the model and just want to apply it to some input data, i.e. we only want to do *forward* computations through the network. We can stop tracking computations by surrounding our computation code with `torch.no_grad()` block:

##### torch.cuda.device_count()

得到目前可用的GPU的数量。

#### DP 与 DDP 的区别

先看 DP，DP 是单进程控制多 GPU。从之前的程序中，我们也可以看出，DP 将输入的一个 batch 数据分成了 n 份（n 为实际使用的 GPU 数量），分别送到对应的 GPU 进行计算。在网络前向传播时，模型会从主 GPU 复制到其它 GPU 上；在反向传播时，每个 GPU 上的梯度汇总到主 GPU 上，求得梯度均值更新模型参数后，再复制到其它 GPU，以此来实现并行。

由于主 GPU 要进行梯度汇总和模型更新，并将计算任务下发给其它 GPU，所以主 GPU 的负载与使用率会比其它 GPU 高，这就导致了 GPU 负载不均衡的现象。

再说说 DDP，DDP 多进程控制多 GPU。系统会为每个 GPU 创建一个进程，不再有主 GPU，每个 GPU 执行相同的任务。DDP 使用分布式数据采样器（DistributedSampler）加载数据，确保数据在各个进程之间没有重叠。在反向传播时，各 GPU 梯度计算完成后，各进程以广播的方式将梯度进行汇总平均，然后每个进程在各自的 GPU 上进行梯度更新，从而确保每个 GPU 上的模型参数始终保持一致。由于无需在不同 GPU 之间复制模型，DPP 的传输数据量更少，因此速度更快。

DistributedDataParallel 既可用于单机多卡也可用于多机多卡，它能够解决 DataParallel 速度慢、GPU 负载不均衡等问题。因此，官方更推荐使用 DistributedDataParallel 来进行分布式训练，也就是接下来要说的 DDP 训练。

##### optimizer.zero_grad()和model.zero_grad()区别

如果1个优化器对应1个模型的时候(1个优化器可以对应多个模型)。optimizer.zero_grad()与model.zero_grad()是相同的。
如果是多个模型的时候，就要具体情况具体分析了。如果直接optimizer.zero_grad()的话，就会把所有模型的梯度清零。

##### optimizer.zero_grad()

optimizer.zero_grad()将梯度归零。