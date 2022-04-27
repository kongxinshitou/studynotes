##### 一行代码实现矩阵转置

```python
li
#[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
[list(_) for _ in zip(*li)]
#[[1, 4, 7], [2, 5, 8], [3, 6, 9]]
#直接转置li矩阵

```

##### BaseException

```python
try:
    pass
except BaseException:
    pass
#一种编码方式
```

##### 计算两天时间间隔

```python
import datetime
a1,b1,c1 = list(map(int,input().split(' ')))
a2,b2,c2 = list(map(int,input().split(' ')))
print((datetime.datetime(a2,b2,c2) - datetime.datetime(a1,b1,c1)).days)
```

##### 反转字符串

```python
s1[::-1]
```

