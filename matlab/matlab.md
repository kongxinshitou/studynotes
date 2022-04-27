##### addpath

向搜索路径中添加文件夹

##### matlab的广播机制

```matlab
>> A=[1 2;3 4]

A =

     1     2
     3     4

>> B = [0 3]

B =

     0     3

>> A - B

ans =

     1    -1
     3     1
```



##### pwd

确定当前文件夹

##### sparse

`S = sparse(A)` 通过挤出任何零元素将满矩阵转换为稀疏格式。如果矩阵包含许多零，将矩阵转换为稀疏存储空间可以节省内存。

`S = sparse(m,n)` 生成 `m`×`n` 全零稀疏矩阵。

`S = sparse(i,j,v)` 根据 `i`、`j` 和 `v` 三元组生成稀疏矩阵 `S`，以便 `S(i(k),j(k)) = v(k)`。`max(i)`×`max(j)` 输出矩阵为 `length(v)` 个非零值元素分配了空间。

如果输入 `i`、`j` 和 `v` 为向量或矩阵，则它们必须具有相同数量的元素。参数 `v` 和/或 `i` 或 `j` 其中一个参数可以使标量。

[](https://www.mathworks.com/help/releases/R2021a/matlab/ref/sparse.html#bul62_s)

`S = sparse(i,j,v,m,n)` 将 `S` 的大小指定为 `m`×`n`。

[](https://www.mathworks.com/help/releases/R2021a/matlab/ref/sparse.html#bul62_o)

`S = sparse(i,j,v,m,n,nz)` 为 `nz` 非零元素分配空间。可以使用此语法为构造后要填充的非零值分配额外空间。

##### find

`k = find(X)` returns a vector containing the [linear indices](https://localhost:31515/static/help/matlab/ref/find.html#buf0c2r-8) of each nonzero element in array `X`.

- If `X` is a vector, then `find` returns a vector with the same orientation as `X`.
- If `X` is a multidimensional array, then `find` returns a column vector of the linear indices of the result.

[](https://localhost:31515/static/help/matlab/ref/find.html#buf0d0b-1)

`k = find(X,n)` returns the first `n` indices corresponding to the nonzero elements in `X`.



`k = find(X,n,direction)`, where `direction` is `'last'`, finds the last `n` indices corresponding to nonzero elements in `X`. The default for `direction` is `'first'`, which finds the first `n` indices corresponding to nonzero elements.

`[row,col] = find(___)` returns the row and column subscripts of each nonzero element in array `X` using any of the input arguments in previous syntaxes.

`[row,col,v] = find(___)` also returns vector `v`, which contains the nonzero elements of `X`.

##### all

`B = all(A)` tests along the first array dimension of `A` whose size does not equal 1, and determines if the elements are all nonzero or logical `1` (`true`). In practice, `all` is a natural extension of the logical AND operator.

- If `A` is a vector, then `all(A)` returns logical `1` (`true`) if all the elements are nonzero and returns logical `0` (`false`) if one or more elements are zero.
- If `A` is a nonempty matrix, then `all(A)` treats the columns of `A` as vectors and returns a row vector of logical `1`s and `0`s.
- If `A` is an empty 0-by-0 matrix, then `all(A)` returns logical `1` (`true`).
- If `A` is a multidimensional array, then `all(A)` acts along the first array dimension whose size does not equal 1 and returns an array of logical values. The size of this dimension becomes `1`, while the sizes of all other dimensions remain the same.

`B = all(A,'all')` tests over all elements of `A`. This syntax is valid for MATLAB® versions R2018b and later.

`B = all(A,dim)` tests elements along dimension `dim`. The `dim` input is a positive integer scalar.

[example](https://localhost:31515/static/help/matlab/ref/all.html#mw_5c60aae3-557f-480b-a38a-a31d927e725e)

`B = all(A,vecdim)` tests elements based on the dimensions specified in the vector `vecdim`. For example, if `A` is a matrix, then `all(A,[1 2])` tests over all elements in `A`, since every element of a matrix is contained in the array slice defined by dimensions 1 and 2.

##### randperm

 `p = randperm(n)` 返回行向量，其中包含从 1 到 `n` 没有重复元素的整数随机排列。

 `p = randperm(n,k)` 返回行向量，其中包含在 1 到 `n` 之间随机选择的 `k` 个唯一整数。

##### length

 `L = length(X)` 返回 `X` 中最大数组维度的长度。对于向量，长度仅仅是元素数量。对于具有更多维度的数据，长度为 `max(size(X))`。空数组的长度为零。

##### norm

 `n = norm(v)` returns the [Euclidean norm](https://localhost:31517/static/help/matlab/ref/norm.html#bvhji30-1) of vector `v`. This norm is also called the 2-norm, vector magnitude, or Euclidean length.

 `n = norm(v,p)` returns the [generalized vector p-norm](https://localhost:31517/static/help/matlab/ref/norm.html#bvhji30-2).

##### std

 `S = std(A)` 返回 `A` 沿大小不等于 1 的第一个数组维度的元素的[标准差](https://www.mathworks.com/help/releases/R2021a/matlab/ref/std.html#bune77u)。

 - 如果 `A` 是观测值的向量，则标准差为标量。
 - 如果 `A` 是一个列为随机变量且行为观测值的矩阵，则 `S` 是一个包含与每列对应的标准差的行向量。
 - 如果 `A` 是多维数组，则 `std(A)` 会沿大小不等于 1 的第一个数组维度计算，并将这些元素视为向量。此维度的大小将变为 `1`，而所有其他维度的大小保持不变。
 - 默认情况下，标准差按 `N-1` 实现归一化，其中 `N` 是观测值数量。

 `S = std(A,w)` 为上述任意语法指定一个权重方案。当 `w = 0` 时（默认值），`S` 按 `N-1` 进行归一化。当 `w = 1` 时，`S` 按观测值数量 `N` 进行归一化。`w` 也可以是包含非负元素的权重向量。在这种情况下，`w` 的长度必须等于 `std` 将作用于的维度的长度。

##### sortrows

 `B = sortrows(A)` 基于第一列中的元素按升序对矩阵行进行排序。当第一列包含重复的元素时，`sortrows` 会根据下一列中的值进行排序，并对后续的相等值重复此行为。

 `B = sortrows(A,column)` 基于向量 `column` 中指定的列对 `A` 进行排序。例如，`sortrows(A,4)` 会基于第四列中的元素按升序对 `A` 的行进行排序。`sortrows(A,[4 6])` 首先基于第四列中的元素，然后基于第六列中的元素，对 `A` 的行进行排序。

##### sort

 `B = sort(A)` 按升序对 `A` 的元素进行排序。

 - 如果 `A` 是向量，则 `sort(A)` 对向量元素进行排序。
 - 如果 `A` 是矩阵，则 `sort(A)` 会将 `A` 的列视为向量并对每列进行排序。
 - 如果 `A` 是多维数组，则 `sort(A)` 会沿大小不等于 1 的第一个数组维度计算，并将这些元素视为向量。

 `B = sort(A,dim)` 返回 `A` 沿维度 `dim` 的排序元素。例如，如果 `A` 是一个矩阵，则 `sort(A,2)` 对每行中的元素进行排序。

 `B = sort(___,direction)` 使用上述任何语法返回按 `direction` 指定的顺序显示的 `A` 的有序元素。`'ascend'` 表示升序（默认值），`'descend'` 表示降序。

 `B = sort(___,Name,Value)` 指定用于排序的其他参数。例如，`sort(A,'ComparisonMethod','abs')` 按模对 `A` 的元素进行排序。

 `[B,I] = sort(___)` 还会为上述任意语法返回一个索引向量的集合。`I` 的大小与 `A` 的大小相同，它描述了 `A` 的元素沿已排序的维度在 `B` 中的排列情况。例如，如果 `A` 是一个向量，则 `B = A(I)`。

##### ismember

如果 `A` 中某位置的数据能在 `B` 中找到，`Lia = ismember(A,B)` 将返回一个在该位置包含逻辑值 `1` (`true`) 的数组。数组中的其他位置将包含逻辑值 `0` (`false`)。

##### eigs

`d = eigs(A)` 返回一个向量，其中包含矩阵 `A` 的六个模最大的特征值。当使用 `eig` 计算所有特征值的计算量很大时（例如对于大型稀疏矩阵来说），这是非常有用的。

`d = eigs(A,k)` 返回 `k` 个模最大的特征值。

`d = eigs(A,k,sigma)` 基于 `sigma` 的值返回 `k` 个特征值。例如，`eigs(A,k,'smallestabs')` 返回 `k` 个模最小的特征值。

`d = eigs(A,k,sigma,Name,Value)` 使用一个或多个名称-值对组参数指定其他选项。例如，`eigs(A,k,sigma,'Tolerance',1e-3)` 将调整算法的收敛容差。

`d = eigs(A,k,sigma,opts)` 使用结构体指定选项。

`d = eigs(A,B,___)` 解算广义特征值问题 `A*V = B*V*D`。您可以选择指定 `k`、`sigma`、`opts` 或名称-值对组作为额外的输入参数。

`d = eigs(Afun,n,___)` 指定函数句柄 `Afun`，而不是矩阵。第二个输入 `n` 可求出 `Afun` 中使用的矩阵 `A` 的大小。您可以选择指定 `B`、`k`、`sigma`、`opts` 或名称-值对组作为额外的输入参数。

`[V,D] = eigs(___)` 返回对角矩阵 `D` 和矩阵 `V`，前者包含主对角线上的特征值，后者的各列中包含对应的特征向量。您可以使用上述语法中的任何输入参数组合。

`[V,D,flag] = eigs(___)` 也返回一个收敛标志。如果 `flag` 为 `0`，则表示已收敛所有特征值。



##### load

 `load(filename)` 从 `filename` 加载数据。

 - 如果 `filename` 是 MAT 文件，`load(filename)` 会将 MAT 文件中的变量加载到 MATLAB® 工作区。
 - 如果 `filename` 是 ASCII 文件，`load(filename)` 会创建一个包含该文件数据的双精度数组。
 - `load(filename,variables)` 加载 MAT 文件 `filename` 中的指定变量。
 - `load(filename,'-ascii')` 将 `filename` 视为 ASCII 文件，而不管文件扩展名如何。
 - `load(filename,'-mat')` 将 `filename` 视为 MAT 文件，而不管文件扩展名如何。

##### floor

 ` Y= floor(X)` 将 `X` 的每个元素舍入到小于或等于该元素的最接近整数。

##### tabulate

 `tabulate(x)` displays a frequency table of the data in the vector `x`. For each unique value in `x`, the `tabulate` function shows the number of instances and percentage of that value 



##### length

 `L = length(X)` 返回 `X` 中最大数组维度的长度。对于向量，长度仅仅是元素数量。对于具有更多维度的数据，长度为 `max(size(X))`。空数组的长度为零。

##### unique

 `C = unique(A)` 返回与 `A` 中相同的数据，但是不包含重复项。`C` 已排序。

##### eye

 `I = eye(n)` returns an `n`-by-`n` identity matrix with ones on the main diagonal and zeros elsewhere.

##### rand

 `X = rand` 返回一个在区间 (0,1) 内均匀分布的随机数。

 `X = rand(n)` 返回一个 `n`×`n` 的随机数矩阵。

 `X = rand(sz1,...,szN)` 返回由随机数组成的 `sz1`×...×`szN` 数组，其中 `sz1,...,szN` 指示每个维度的大小。例如：`rand(3,4)` 返回一个 3×4 的矩阵。

 `X = rand(sz)` 返回由随机数组成的数组，其中大小向量 `sz` 指定 `size(X)`。例如：`rand([3 4])` 返回一个 3×4 的矩阵。

 `X = rand(___,typename)` 返回由 `typename` 数据类型的随机数组成的数组。`typename` 输入可以是 `'single'` 或 `'double'`。您可以使用上述语法中的任何输入参数。

 `X = rand(___,'like',p)` 返回由 `p` 等随机数组成的数组；也就是与 `p` 同一对象类型。您可以指定 `typename` 或 `'like'`，但不能同时指定两者。

 `X = rand(s,___)` 从随机数流 `s` 而不是默认全局流生成数字。要创建一个流，请使用 [`RandStream`](https://www.mathworks.com/help/releases/R2021a/matlab/ref/randstream.html)。指定 `s`，后跟上述语法中的任意参数组合，但涉及 `'like'` 的组合除外。此语法不支持 `'like'` 输入

##### sum

 `S = sum(A)` 返回 A 沿大小不等于 1 的第一个数组维度的元素之和。

 - 如果 `A` 是向量，则 `sum(A)` 返回元素之和。
 - 如果 `A` 是矩阵，则 `sum(A)` 将返回包含每列总和的行向量。
 - 如果 `A` 是多维数组，则 `sum(A)` 沿大小不等于 1 的第一个数组维度计算，并将这些元素视为向量。此维度会变为 `1`，而所有其他维度的大小保持不变。

 `S = sum(A,'all')` 计算 `A` 的所有元素的总和。此语法适用于 MATLAB® R2018b 及更高版本。

 `S = sum(A,dim)` 沿维度 `dim` 返回总和。例如，如果 `A` 为矩阵，则 `sum(A,2)` 是包含每一行总和的列向量。

 `S = sum(A,vecdim)` 根据向量 `vecdim` 中指定的维度对 `A` 的元素求和。例如，如果 `A` 是矩阵，则 `sum(A,[1 2])` 是 `A` 中所有元素的总和，因为矩阵的每个元素包含在由维度 1 和 2 定义的数组切片中。

##### repmat

`B = repmat(A,n)` 返回一个数组，该数组在其行维度和列维度包含 `A` 的 `n` 个副本。`A` 为矩阵时，`B` 大小为 `size(A)*n`。

`B = repmat(A,r1,...,rN)` 指定一个标量列表 `r1,..,rN`，这些标量用于描述 `A` 的副本在每个维度中如何排列。当 `A` 具有 `N` 维时，`B` 的大小为 `size(A).*[r1...rN]`。例如：`repmat([1 2; 3 4],2,3)` 返回一个 4×6 的矩阵。

`B = repmat(A,r)` 使用行向量 `r` 指定重复方案。例如，`repmat(A,[2 3])` 与 `repmat(A,2,3)` 返回相同的结果。

##### max

`M = max(A)` 返回数组的最大元素。

- 如果 `A` 是向量，则 `max(A)` 返回 `A` 的最大值。
- 如果 `A` 为矩阵，则 `max(A)` 是包含每一列的最大值的行向量。
- 如果 `A` 是多维数组，则 `max(A)` 沿大小不等于 `1` 的第一个数组维度计算，并将这些元素视为向量。此维度的大小将变为 `1`，而所有其他维度的大小保持不变。如果 `A` 是第一个维度长度为零的空数组，则 `max(A)` 返回与 `A` 大小相同的空数组。

`M = max(A,[],dim)` 返回维度 `dim` 上的最大元素。例如，如果 `A` 为矩阵，则 `max(A,[],2)` 是包含每一行的最大值的列向量。

`M = max(A,[],nanflag)` 指定在计算中包括还是忽略 `NaN` 值。例如，`max(A,[],'includenan')` 包括 `A` 中的所有 `NaN` 值，而 `max(A,[],'omitnan')` 忽略这些值。

`M = max(A,[],dim,nanflag)` 还指定使用 `nanflag` 选项时的运算维度。

`[M,I] = max(___)` 在上述语法基础上，还返回 `A` 中最大值在运算维度上的对应索引。

`M = max(A,[],'all')` 查找 `A` 的所有元素的最大值。此语法适用于 MATLAB® R2018b 及更高版本。

`M = max(A,[],vecdim)` 计算向量 `vecdim` 所指定的维度上的最大值。例如，如果 `A` 是矩阵，则 `max(A,[],[1 2])` 计算 `A` 中所有元素的最大值，因为矩阵的每个元素都包含在由维度 1 和 2 定义的数组切片中。

`M = max(A,[],'all',nanflag)` 计算在使用 `nanflag` 选项时 `A` 的所有元素的最大值。

`M = max(A,[],vecdim,nanflag)` 指定在使用 `nanflag` 选项时要运算的多个维度。

`[M,I] = max(A,[],___,'linear')` 返回 `A` 中最大值在 `A` 中的对应线性索引。

`C = max(A,B)` 返回从 `A` 或 `B` 中提取的最大元素的数组。

`C = max(A,B,nanflag)` 还指定如何处理 `NaN` 值。

##### svd

`s = svd(A)` 以降序顺序返回矩阵 `A` 的[奇异值](https://www.mathworks.com/help/releases/R2021a/matlab/math/singular-values.html)。

`[U,S,V] = svd(A)` 执行矩阵 `A` 的奇异值分解，因此 `A = U*S*V'`。

`[U,S,V] = svd(A,'econ')` 为 `m`×`n` 矩阵 `A` 生成精简分解：

- `m > n` - 只计算 `U` 的前 `n` 列，`S` 是一个 `n`×`n` 矩阵。
- `m = n` - `svd(A,'econ')` 等效于 `svd(A)`。
- `m < n` - 只计算 `V` 的前 `m` 列，`S` 是一个 `m`×`m` 矩阵。

精简分解从奇异值的对角矩阵 `S` 中删除额外的零值行或列，以及 `U` 或 `V` 中与表达式 `A = U*S*V'` 中的那些零值相乘的列。删除这些零值和列可以缩短执行时间，并减少存储要求，而且不会影响分解的准确性。

[](https://www.mathworks.com/help/releases/R2021a/matlab/ref/double.svd.html#bu2_4r8)

`[U,S,V] = svd(A,0)` 为 `m`×`n` 矩阵 `A` 生成另一种精简分解：

- `m > n` - `svd(A,0)` 等效于 `svd(A,'econ')`。
- `m <= n` - `svd(A,0)` 等效于 `svd(A)`。

##### ones

 `X = ones` 返回标量 `1`。

 `X = ones(n)` 返回一个 `n`×`n` 的全 1 矩阵。

 `X = ones(sz1,...,szN)` 返回由 1 组成的 `sz1`×...×`szN` 数组，其中 `sz1,...,szN` 指示每个维度的大小。例如，`ones(2,3)` 返回由 1 组成的 2×3 数组。

 `X = ones(sz)` 返回一个由 1 组成的数组，其中大小向量 `sz` 定义 `size(X)`。例如，`ones([2,3])` 返回由 1 组成的 2×3 数组。

 `X = ones(___,typename)` 在上述语法的基础上进一步指定 `X` 的数据类型（类）。例如，`ones(5,'int8')` 返回 8 位整数组成的 5×5 矩阵。

 `X = ones(___,'like',p)` 指定 `X` 具有与数值变量 `p` 相同的数据类型、稀疏性和复/实性（实数或复数）。

##### trace

 `b = trace(A)` calculates the sum of the diagonal elements of matrix `A`:

###### linspace

 `y = linspace(x1,x2,n)` 生成 `n` 个点。这些点的间距为 `(x2-x1)/(n-1)`。

##### mean

`M = mean(A)` 返回 `A` 沿大小不等于 1 的第一个数组维度的元素的[均值](https://www.mathworks.com/help/releases/R2021a/matlab/ref/mean.html#bupom9u)。

- 如果 `A` 是向量，则 `mean(A)` 返回元素均值。
- 如果 `A` 为矩阵，那么 `mean(A)` 返回包含每列均值的行向量。
- 如果 `A` 是多维数组，则 `mean(A)` 沿大小不等于 1 的第一个数组维度计算，并将这些元素视为向量。此维度会变为 `1`，而所有其他维度的大小保持不变。

`M = mean(A,'all')` 计算 `A` 的所有元素的均值。此语法适用于 MATLAB® R2018b 及更高版本。

`M = mean(A,dim)` 返回维度 `dim` 上的均值。例如，如果 `A` 为矩阵，则 `mean(A,2)` 是包含每一行均值的列向量。

`M = mean(A,vecdim)` 计算向量 `vecdim` 所指定的维度上的均值。例如，如果 `A` 是矩阵，则 `mean(A,[1 2])` 是 `A` 中所有元素的均值，因为矩阵的每个元素都包含在由维度 1 和 2 定义的数组切片中。

`M = mean(___,outtype)` 使用前面语法中的任何输入参数返回指定的数据类型的均值。`outtype` 可以是 `'default'`、`'double'` 或 `'native'`。

`M = mean(___,nanflag)` 指定在上述任意语法的计算中包括还是忽略 `NaN` 值。`mean(A,'includenan')` 会在计算中包括所有 `NaN` 值，而 `mean(A,'omitnan')` 则忽略这些值。

##### 注释代码快捷键

 Ctrl+R注释一段代码，Ctrl+T反注释代码

#### normrnd

 `r = normrnd(mu,sigma)` 从均值参数为 `mu` 和标准差参数为 `sigma` 的正态分布中生成随机数。

 `r = normrnd(mu,sigma,sz1,...,szN)` 生成正态随机数数组，其中 `sz1,...,szN` 指示每个维度的大小。

 `r = normrnd(mu,sigma,sz)` 生成正态随机数数组，其中向量 `sz` 指定 `size(r)`。

#### mvnrnd

`R = mvnrnd(mu,Sigma,n)` returns a matrix `R` of `n` random vectors chosen from the same multivariate normal distribution, with mean vector `mu` and covariance matrix `Sigma`. 
`R = mvnrnd(mu,Sigma)` returns an *m*-by-*d* matrix `R` of random vectors sampled from *m* separate *d*-dimensional multivariate normal distributions, with means and covariances specified by `mu` and `Sigma`, respectively. Each row of `R` is a single multivariate normal random vector.

#### fullfile

`f = fullfile(filepart1,...,filepartN)` 根据指定的文件夹和文件名构建完整的文件设定。`fullfile` 在必要情况下插入依平台而定的文件分隔符，但不添加尾随的文件分隔符。在 Windows® 平台上，文件分隔符为反斜杠 (`\`)。在其他平台上，文件分隔符可能为不同字符。

#### dir

`dir` 列出当前文件夹中的文件和文件夹。

`dir name` 列出与 `name` 匹配的文件和文件夹。如果 `name` 为文件夹，`dir` 列出该文件夹的内容。使用绝对或相对路径名称指定 `name`。`name` 参数的文件名可以包含 `*` 通配符，路径名称可以包含 `*` 和 `**` 通配符。与 `**` 通配符相邻的字符必须为文件分隔符。

`listing = dir(name)` 返回 `name` 的属性

#### strrep

`newStr = strrep(str,old,new)` 将 `str` 中出现的所有 `old` 都替换为 `new`。

##### pinv

`B = pinv(A)` 返回矩阵 `A` 的 [Moore-Penrose 伪逆](https://www.mathworks.com/help/releases/R2021a/matlab/ref/pinv.html#mw_ffa95973-29a2-48a1-adb0-5a4214e0d9cf)。

`B = pinv(A,tol)` 指定容差的值。`pinv` 将 `A` 中小于容差的奇异值视为零。