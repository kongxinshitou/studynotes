#### Python语法

##### **kwargs

* 传递可变参数字典

##### 浅拷贝

* 只拷贝"第一层"，第一层之后的不管

##### 深拷贝

* 拷贝后几层

##### 字符串前面+''

在Python的string前面加上‘r’， 是为了告诉编译器这个string是个raw string，不要转意backslash '\' 。 例如，\n 在raw string中，是两个字符，\和n， 而不会转意为换行符。由于正则表达式和 \ 会有冲突，因此，当一个字符串使用了正则表达式后，最好在前面加上'r'。

##### nonlocal

往上面嵌套的层去找，直到找到

##### 匿名函数

```python
lambda x: x * x
```

等价于

```python
    return x * x
```

* 关键字`lambda`表示匿名函数，冒号前面的`x`表示函数参数。
* 匿名函数有个限制，就是只能有一个表达式，不用写`return`，返回值就是该表达式的结果。
* 用匿名函数有个好处，因为函数没有名字，不必担心函数名冲突。此外，匿名函数也是一个函数对象，也可以把匿名函数赋值给一个变量，再利用变量来调用该函数。

##### round()

* **round()** 方法返回浮点数x的四舍五入值。
  * x -- 数值表达式。
  * n -- 数值表达式，表示从小数点位数。

```python
round( x [, n]  )
```

##### re.search

```python
re.search(pattern, string, flags=0)
#扫描整个 字符串 找到匹配样式的第一个位置，并返回一个相应的 匹配对象。如果没有匹配，就返回一个 None ； 注意这和找到一个零长度匹配是不同的。
```

##### os.splitext

```python
os.path.splitext(path)
#将路径名称 path 拆分为 (root, ext) 对使得 root + ext == path，并且扩展名 ext 为空或以句点打头并最多只包含一个句点。如果路径 path 不包含扩展名，则 ext 将为 ''
如果路径 path 包含扩展名，则 ext 将被设为该扩展名，包括打头的句点。 请注意在其之前的句点将被忽略:
>>> splitext('foo.bar.exe')
('foo.bar', '.exe')
>>> splitext('/foo/bar.exe')
('/foo/bar', '.exe')
```



##### os.listdir

```python
os.listdir(path='.')
#返回一个包含由 path 指定目录中条目名称组成的列表。 该列表按任意顺序排列，并且不包括特殊条目 '.' 和 '..'，即使它们存在于目录中。 如果有文件在调用此函数期间在被移除或添加到目录中，是否要包括该文件的名称并没有规定。
```

##### re.match

```python
re.match(pattern, string, flags=0)
#如果 string 开始的0或者多个字符匹配到了正则表达式样式，就返回一个相应的 匹配对象 。 如果没有匹配，就返回 None ；注意：它只匹配开始的比如 'sb' 匹配'sbwo',不匹配'wosb'

```

如果 *string* 开始的0或者多个字符匹配到了正则表达式样式，就返回一个相应的 [匹配对象](https://docs.python.org/zh-cn/3/library/re.html#match-objects) 。 如果没有匹配，就返回 `None` ；

注意：它只匹配开始的比如 'sb' 匹配'sbwo',不匹配'wosb'

##### os.walk

```python
os.walk(top, topdown=True, onerror=None, followlinks=False)
```

生成目录树中的文件名，方式是按上->下或下->上顺序浏览目录树。对于以 *top* 为根的目录树中的每个目录（包括 *top* 本身），它都会生成一个三元组 `(dirpath, dirnames, filenames)`。

*dirpath* 是表示目录路径的字符串。 *dirnames* 是 *dirpath* 中子目录名称组成的列表 (excluding `'.'` and `'..'`)。 *filenames* 是 *dirpath* 中非目录文件名称组成的列表。 请注意列表中的名称不带路径部分。 要获取 *dirpath* 中文件或目录的完整路径（以 *top* 打头），请执行 `os.path.join(dirpath, name)`。 列表是否排序取决于具体文件系统。 如果有文件或列表生成期间被移除或添加到 *dirpath* 目录中，是否要包括该文件的名称并没有规定。

如果可选参数 *topdown* 为 `True` 或未指定，则在所有子目录的三元组之前生成父目录的三元组（目录是自上而下生成的）。如果 *topdown* 为 `False`，则在所有子目录的三元组生成之后再生成父目录的三元组（目录是自下而上生成的）。无论 *topdown* 为何值，在生成目录及其子目录的元组之前，都将检索全部子目录列表。

当 *topdown* 为 `True` 时，调用者可以就地修改 *dirnames* 列表（也许用到了 [`del`](https://docs.python.org/zh-cn/3/reference/simple_stmts.html#del) 或切片），而 [`walk()`](https://docs.python.org/zh-cn/3/library/os.html?highlight=walk#os.walk) 将仅仅递归到仍保留在 *dirnames* 中的子目录内。这可用于减少搜索、加入特定的访问顺序，甚至可在继续 [`walk()`](https://docs.python.org/zh-cn/3/library/os.html?highlight=walk#os.walk) 之前告知 [`walk()`](https://docs.python.org/zh-cn/3/library/os.html?highlight=walk#os.walk) 由调用者新建或重命名的目录的信息。当 *topdown* 为 `False` 时，修改 *dirnames* 对 walk 的行为没有影响，因为在自下而上模式中，*dirnames* 中的目录是在 *dirpath* 本身之前生成的。

默认将忽略 [`scandir()`](https://docs.python.org/zh-cn/3/library/os.html?highlight=walk#os.scandir) 调用中的错误。如果指定了可选参数 *onerror*，它应该是一个函数。出错时它会被调用，参数是一个 [`OSError`](https://docs.python.org/zh-cn/3/library/exceptions.html#OSError) 实例。它可以报告错误然后继续遍历，或者抛出异常然后中止遍历。注意，可以从异常对象的 `filename` 属性中获取出错的文件名。

##### getattr()

* **getattr()** 函数用于返回一个对象属性值。
  * object -- 对象。
  * name -- 字符串，对象属性。
  * default -- 默认返回值，如果不提供该参数，在没有对应属性时，将触发 AttributeError。

```python
getattr(object, name[, default])
```

##### seq[i:j:k]

```Python
#片第从i到j与第k步
>>> range(10)[::2]
[0, 2, 4, 6, 8]
```

##### //

```python
// 称为地板除，两个整数的除法仍然是整数，它总是会舍去小数部分，返回数字序列中比真正的商小的，最接近的数字。
```

##### zip()

```python
#zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
```

##### 在Python中一维数组不分行列。

##### sort 与 sorted 区别

* sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
* list 的 sort 方法返回的是对已经存在的列表进行操作，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。

##### 惰性对象

* [* enumerate(A)] #用来打开惰性对象:enumerate,map,zip

#### numpy语法

##### argmin

```python
numpy.argmin(a, axis=None, out=None, *, keepdims=<no value>)[source]
Returns the indices of the minimum values along an axis.
```

##### np.random.random

```python

#Return random floats in the half-open interval [0.0, 1.0). Alias for random_sample to ease forward-porting to the new random API.
np.random.random((10,10))
#
```

##### np.random.randn

```python
random.randn(d0, d1, ..., dn)
#Return a sample (or samples) from the “standard normal” distribution.
```



##### norm

```python
np.linalg.norm(x, ord=None, axis=None, keepdims=False)[source]
Matrix or vector norm.

#This function is able to return one of eight different matrix norms, or one of an infinite number of vector norms (described below), depending on the value of the ord parameter.
```

##### np.any()

```python
numpy.any(a, axis=None, out=None, keepdims=<no value>, **, *where=<no value>)

#Test whether any array element along a given axis evaluates to True.Returns single boolean unless axis is not None
a = np.array([[1,2],[0,1]])
np.any(a==0)
#判断矩阵中是否有0元素
```

##### np.all()

```python
numpy.all(a, axis=None, out=None, keepdims=<no value>, *, where=<no value>)[source]
#Test whether all array elements along a given axis evaluate to True.
a = np.array([[1,2],[0,1]])
np.all(a==0)
#判断矩阵中是否全是0

```



##### random.randint

```python
random.randint(low, high=None, size=None, dtype=int)
Return random integers from low (inclusive) to high (exclusive).
```

##### reshape(-1,n)

- 确定要n列，行数由计算机决定

##### ravel()

* ravel()将二维数组按行降维成一维

##### 对数组进行深复制

```python
d = a.copy()  # a new array object with new data is created
```



##### meshgrid和vstack

* meshgrid 函数用来生成网格矩阵，可以是二维网格矩阵
* vstack将多行堆叠到一起

```python
a=np.array([1,2,3])
b=np.array([4,5,6,6])
P,Q=np.meshgrid(a,b)
P
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])
Q
array([[4, 4, 4],
       [5, 5, 5],
       [6, 6, 6],
       [6, 6, 6]])
K=np.vstack([Q.ravel(),P.ravel()])
array([[4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6],
       [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]])
```

##### unique()

* 数组去重并返回去重后的结果。

##### logspace(-2,10,10)

* 形成对数间隔均匀的10个数。

```python
np.logspace(-2,10,10)
array([1.00000000e-02, 2.15443469e-01, 4.64158883e+00, 1.00000000e+02,
       2.15443469e+03, 4.64158883e+04, 1.00000000e+06, 2.15443469e+07,
       4.64158883e+08, 1.00000000e+10])
```

##### shuffle(A)

* 对A list洗牌，打乱顺序

#### pandas语法

##### pd.sina

```python
pd.isna(Dataframe)
#判断Dataframe是否有空值，并返回一个Dataframe
```



##### series

Pandas Series 类似表格中的一个列（column），类似于一维数组，可以保存任何数据类型。

Series 由索引（index）和列组成，函数如下：

```
pandas.Series( data, index, dtype, name, copy)
```

参数说明：

- **data**：一组数据(ndarray 类型)。
- **index**：数据索引标签，如果不指定，默认从 0 开始。
- **dtype**：数据类型，默认会自己判断。
- **name**：设置名称。
- **copy**：拷贝数据，默认为 False。

##### DataFrame

DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。

DataFrame 构造方法如下：

```
pandas.DataFrame( data, index, columns, dtype, copy)
```

参数说明：

- **data**：一组数据(ndarray、series, map, lists, dict 等类型)。
- **index**：索引值，或者可以称为行标签。
- **columns**：列标签，默认为 RangeIndex (0, 1, 2, …, n) 。
- **dtype**：数据类型。
- **copy**：拷贝数据，默认为 False。

##### Series.map(dict)

* 根据字典的键值对，替换Series里面的值，如果字典中，没有找到Series中对应的值，则赋值为NaN

##### read_csv

* 读csv文件,返回一个DataFrame

```python
weather = pd.read_csv(r"D:\BaiduNetdiskDownload\MLcaicai\svm2\weatherAUS5000.csv",
                     index_col=0)#index_col是索引所在列好
```

##### DataFrame.info()

* 查看数据的统计信息

##### DataFrame.isnull()

* 判断是否是缺失值,如果是缺失值则返回相应的条目表示为True，否则表示为False

##### DataFrame.to_csv("文件名.csv")

* 将DataFrame保存到csv文件中

##### DataFrame.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99])

* 对数据进行描述性统计

##### DataFrame.value_counts(X)

* 统计X中的每一个条目出现的次数

##### DataFrame.sort_values(by="location")

* 根据某一个特征来对数据进行排序

#### apply()

* apply是对DataFrame上的某一列进行处理的一个函数
* lambda x匿名函数，请在dataframe上这一列中的每一行帮我执行冒号后的命令

```python
Xtrain["Date"] = Xtrain["Date"].apply(lambda x:int(x.split("-")[1]))
```

##### 修改DataFrame列名

* 通常我们使用df.columns = 某个列表 这样的形式来一次修改所有的列名

##### 切片

* 通过df.iloc来代替np的一些切片操作

#### sklearn

##### make_blobs

* make_blobs函数是为聚类产生一个数据集和相应的标签
* n_samples:表示数据样本点个数,默认值100
* n_features:表示数据的维度，默认值是2
* centers:产生数据的中心点，默认值3
* cluster_std：数据集的标准差，浮点数或者浮点数序列，默认值1.0
* center_box：中心确定之后的数据边界，默认值(-10.0, 10.0)
* shuffle ：洗乱，默认值是True

#### JupyterNotebook技巧

##### 查看函数参数

* 鼠标点进括号，然后shift+tb

##### 快捷键

* 分裂的快捷键：ctrl shitf -
* 合并的快捷键 shift M
* 在上方添加一个新的cell ESC a enter
* 在下方添加一个新的cell ESC b enter
* 删除一个cell ESC d d
* ctrl和鼠标滚轮可以用来控制页面缩放
* ESC +i i中断某个cell运行

#### random

##### uniform

* 从输入的任意两个整数中取出size个随机数

```python
X = rnd.uniform(-3, 3, size=100) 
```

##### normal

* 生成size个服从正态分布的随机数

```python 
rnd.normal(size=len(X))
```



