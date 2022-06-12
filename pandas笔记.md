### Series 
Series是一个保存一维数组的对象，这个对象的数据表现有两列，一列是轴标签，一列是保存的一维数组(ndarray)
##### Series的创建
###### Series 创建方式一
```python
import pandas as pd
obj=pd.Series([4,7,-5,3])  
print(obj)
#输出：
# 0    4
# 1    7
# 2   -5
# 3    3
#dtype: int64

'''查看索引与值'''
print(obj.values) #查看数据列 [ 4  7 -5  3]
print(obj.index) #RangeIndex(start=0, stop=4, step=1)   是一个迭代器
print(list(obj.index))# 查看索引列 [0, 1, 2, 3]
print(type(obj.values)) # 一维数组 <class 'numpy.ndarray'>
print(type(obj.index)) # <class 'pandas.core.indexes.range.RangeIndex'>
```
###### Series 创建方式二（指定索引）
```python
import pandas as pd
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 1 ]) 
'''
还可以通过指定索引来创建，要求：
1.长度一定要与数据相同（不然报错）
2. 可重复
3.支持python各种数据类型，建议使用字符串和整型
'''
```
###### Series 创建方式三（字典）
```python 
import pandas as pd
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
```
###### Series 创建方式四（使用numpy数组+自定义索引）
```python
import pandas as pd
import numpy as np
obj4 = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'a'])
```
###### Series 创建方式五（使用已有的Series对象创建）
```python
obj1 = pd.Series([4, 7, -5, 3], index=['a', 'b', 'c', 'd'])
#输出：
# a    4
# b    7
# c   -5
# d    3
# dtype: int64
obj2 = pd.Series(obj1) #与obj1相同
obj3 = pd.Series(obj1, index=['b', 'a', 'c'])# index在obj1存在就使用obj1的值，否则就是NaN
#输出：
# b    7
# a    4
# c   -5
# dtype: int64    #b,a,c对应的都是obj1的值
obj4 = pd.Series(obj1, index=[0, 1, 3, 7, 9, 0])
#输出：
# 0   NaN
# 1   NaN
# 3   NaN
# 7   NaN
# 9   NaN
# 0   NaN
# dtype: float64
```

##### Series的索引
###### 一般索引
```python
import pandas as pd
import numpy as np
obj4 = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'a'])
print(obj4)
#输出：
# a    0.0
# b    1.0
# c    2.0
# a    3.0

'''使用索引号索引'''
print(obj4[0]) #0.0
print(obj4[1]) #1.0
'''使用轴标签索引'''
print(type(obj4['a'])) # <class 'pandas.core.series.Series'>
print(obj4['a']) 
#输出（如果有重复的轴标签，使用该标签索引时，返回一个Series对象，否则是单个值）：
# a    0.0
# a    3.0
# dtype: float64
print(type(obj4['b'])) # <class 'numpy.float64'>
print(obj4['c']) #2.0

'''轴标签有数值型会与索引号冲突，此时程序会优先选择轴标签进行索引。如果要进行索引号索引，可以用： 变量名.iloc[索引号]'''
obj6 = pd.Series([4, 7, -5, 3], index=['1', '2', 4, 4])
ic(obj6)
print(list(obj6.index))
# print(obj6[0]) # KeyError
print(obj6.iloc[0])
# print(obj6[1]) # KeyError 正确的应该是'1'
print(obj6['2'])
print(obj6[4]) # 或者 print(obj6.loc[4])
```
小结（索引操作）：
1. 变量名[ ] ：如果index没有数值型，可以通过索引号索引，也可以通过轴标签索引，否则只能通过轴标签索引
2. 变量名.iloc[索引号]
3. 变量名.loc[轴标签]

###### 布尔索引
```python
obj1 = pd.Series([4, 7, -5, 3], index=['a', 'b', 'c', 'e'])
condition=obj1>0
print(obj1[condition]) #输出符合条件的索引
print(obj1[obj1>0]) #效果同上
```
###### apply应用方法[将指定的操作(functional)应用到每一个元素]
```python
obj1 = pd.Series([4, 7, -5, 3], index=['a', 'b', 'c', 'e'])
obj2=obj1.apply(lambda x:x**2)
print(obj2)
```

##### Series的切片（返回的是series对象）
```python
'''索引号切片'''
obj1 = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
print(obj1[1:3])# 不包含结束的索引号
#输出：
# b    1.0
# c    2.0
# dtype: float64
print(obj1[0:4])
print(obj1[:4])
print(obj1[0:])
print(obj1[1:6])# 索引号越界也没关系

'''轴标签切片'''
print(obj1['a':'c'])# 结束位置'c'包含在内
#输出：
# a    0.0
# b    1.0
# c    2.0
# dtype: float64
print(obj1['a':]) # 
print(obj1[:'c']) # 结束位置'c'包含在内

'''轴标签有重复时的切片'''
obj2 = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'a'])
print(obj2['b':'c']) #没有歧义就可以  
# print(obj2['a':]) # non-unique label: 'a'（a是重复的没法切片）
print(obj2[:'c']) # 从头开始切片

'''轴标签存在数值型时的切片（如果用数值型进行索引，程序优先选择索引号进行索引，左闭右开）'''
obj3 = pd.Series(np.arange(5.), index=[1, 'b', 'c', 2, 0])
print(obj3)
print(obj3[0:1])
print(obj3[0:4:2])# 步长为2
print(obj3[:2])
print(obj3['b':'c'])

print(obj3.iloc[0:1])
print(obj3.loc[1:0])

```
##### Series的运算（以加法为例，相减和乘除同理）
```python
'''索引没有重复'''
obj1 = pd.Series([4, 7, -5, 3], index=['a', 'b', 'c', 'd'])
obj2 = pd.Series(obj1) 
s=obj1+obj2 
print(s)
#输出（索引对应位置数值相加）：
# a     8
# b    14
# c   -10
# d     6
# dtype: int64

'''索引有重复'''
#1）两个同样的Series相加
obj1 = pd.Series([4, 7, -5, 3], index=['a', 'b', 'c', 'a'])
obj3=pd.Series(obj1, index=[ 'b','a', 'c', 'a'])
s=obj1+obj2 
print(s)
#输出(还是相同位置相加)：
# a     8
# b    14
# c   -10
# a     6
# dtype: int64

#2）如果一方有一个另一方有多个，则分别相加
obj1 = pd.Series([4, 7, -5, 3], index=['a', 'b', 'c', 'a'])
# a    4
# b    7
# c   -5
# a    3
# dtype: int64
obj2=pd.Series([4,7,0,1], index=['b','c','a','b'])
# b    4
# c    7
# a    0
# b    1
# dtype: int64
s=obj1+obj2
print(s)
#输出：
# a     4
# a     3
# b    11
# b     8
# c     2
# dtype: int64

'''索引不对齐(不存在的数值会被NaN代替)'''
obj1 = pd.Series([4, 7, -5, 3], index=['a', 'b', 'c', 'e'])
obj2=pd.Series(obj1, index=['b','c'])
s=obj1+obj2
print(s)
#输出：
# a     NaN
# b    14.0
# c   -10.0
# e     NaN
# dtype: float64

obj1 = pd.Series([4, 7, -5, 3], index=['a', 'b', 'c', 'e'])
obj2=obj2=pd.Series([4,7,0,1], index=['d','f','g','h'])
s=obj1+obj2
print(s)
#输出：
# a   NaN
# b   NaN
# c   NaN
# d   NaN
# e   NaN
# f   NaN
# g   NaN
# h   NaN
# dtype: float64
```


### DataFrame
##### DataFrame的创建
```python
import pandas as pd
import numpy as np
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],'year': [2000, 2001, 2002, 2001, 2002, 2003],'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]} #字典的键是列的索引,value是字典里的值（每一列arrays元素个数必须一样）
frame=pd.DataFrame(data)
print(frame)
#输出：
#     state  year  pop
# 0    Ohio  2000  1.5
# 1    Ohio  2001  1.7
# 2    Ohio  2002  3.6
# 3  Nevada  2001  2.4
# 4  Nevada  2002  2.9
# 5  Nevada  2003  3.2

'''可以指定行(columns)的索引号'''
frame = pd.DataFrame(data, index=['a', 'b', 'c', 'd', 'e', 'f'], columns=['year', 'pop', 'state'])
print(frame)
#输出：
#    year  pop   state
# a  2000  1.5    Ohio
# b  2001  1.7    Ohio
# c  2002  3.6    Ohio
# d  2001  2.4  Nevada
# e  2002  2.9  Nevada
# f  2003  3.2  Nevada

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],'year': [2000, 2001, 2002, 2001, 2002, 2003],'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]} 
frame=pd.DataFrame(data,index=['a','b','c','d','e','f'],columns=[0,1,'year'])
print(frame)
#输出(columns列中1、2都不存在，所以所有值都为NaN)：
#      0    1  year
# a  NaN  NaN  2000
# b  NaN  NaN  2001
# c  NaN  NaN  2002
# d  NaN  NaN  2001
# e  NaN  NaN  2002
# f  NaN  NaN  2003

'''将Series转化为DataFrame,使用Series.to_frame()'''
```
##### DataFrame修改属性
```python
frame = pd.DataFrame(data, index=['a', 'b', 'c', 'd', 'e', 'f'], columns=['year', 'pop', 'state'])

frame.columns = [0, 1, 2] #修改列的索引
frame.index = [0, 1, 2, 3, 4, 5] #修改行的索引
print(frame.dtypes) #查看数据类型
```
##### DataFrame的索引与切片
```python
'''默认索引的是列，所以索引行不存在会报错'''
frame = pd.DataFrame(data, columns=['year', 'pop', 'state'])
#print(frame[0])# KeyError: 0
print(frame['year'])

frame2 = pd.DataFrame(np.arange(9).reshape(3, 3))
#    0  1  2
# 0  0  1  2
# 1  3  4  5
# 2  6  7  8
frame2.index = ['a', 'b', 'c']
#    0  1  2
# a  0  1  2
# b  3  4  5
# c  6  7  8
print(frame2[0])
#输出：
# a    0
# b    3
# c    6
# Name: 0, dtype: int32

'''使用索引号进行行和列的索引可以用frame.iloc[索引号]   （没有指定默认选择行索引）'''
frame2 = pd.DataFrame(data, columns=['year', 'pop', 'state'])
#    year  pop   state
# 0  2000  1.5    Ohio
# 1  2001  1.7    Ohio
# 2  2002  3.6    Ohio
# 3  2001  2.4  Nevada
# 4  2002  2.9  Nevada
# 5  2003  3.2  Nevada
print(frame2.iloc[0])# 行索引
#输出：
# 5  2003  3.2  Nevada
# year     2000
# pop       1.5
# state    Ohio
# Name: 0, dtype: object
print(frame2.iloc[0, 1])#输出：1.5（行索引号0，列索引号为1）
print(frame2.iloc[[0, 1], [0, 1]]) # 数组索引，行索引号为[0, 1]，列索引号[0, 1]
print(frame2.iloc[[0, 1], 1])
#输出：
# 0    1.5
# 1    1.7


frame2 = pd.DataFrame(np.arange(9).reshape(3, 3))
#    0  1  2
# 0  0  1  2
# 1  3  4  5
# 2  6  7  8
print(frame2 > 3)
#输出：
#        0      1      2
# 0  False  False  False
# 1  False   True   True
# 2   True   True   True
print(frame2.iloc[[False, False, True], [True, True, False]])  #True代表索引这一行/列，False代表不索引这一行/列
#输出：
#    0  1
# 2  6  7

'''iloc切片'''
print(frame2.iloc[0:2])#切片
#输出：
#    0  1  2
# 0  0  1  2
# 1  3  4  5
print(frame2.iloc[0:2, 1:])
#输出：
#    1  2
# 0  1  2
# 1  4  5
'''不用iloc默认对列进行标签切片'''
frame2 = pd.DataFrame(np.arange(9).reshape(3, 3))
frame2.index = ['a', 'b', 'c']
print(frame2)
print(frame2[1])# 列标签索引
print(frame2[3:])# 支持索引号切片和标签切片，(行切片)
'''使用标签索引和切片'''
print(frame2.loc['b':'c', [1, 2]])
print(frame2.loc[['a', 'c'], [1, 2]])
```
小结：
1.变量名[ ]
1.1 索引，操作的是列
1.2 切片，操作的是行，（支持索引号和标签）

2.变量名.iloc[ ]，根据索引号进行多轴的操作，用法与np基本相同，不同：数组索引
3.变量名.loc[ ] ,根据标签进行多轴的操作，用法与np基本相同，不同：数组索引

###### DataFrame索引练习
```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],'year': [2000, 2001, 2002, 2001, 2002, 2003],'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]} 
frame = pd.DataFrame(data, index=['a', 'b', 'c', 'd', 'e', 'f'])
# 使用3种方法检索pop列
print(frame['pop'])
print(frame.pop)
print(frame.loc[:, 'pop'])
#同学答案：
print(frame.iloc[:, 2:])
print(frame.iloc[:,2])
print(frame.iloc[:,[False,False,True]]) #布尔索引


'''
data为：
   城市   薪资    房价    人数
0  北京   10000  54670  200000
1  上海    9800  48060  180000
2  深圳   11000  47000  150000
筛选出房价大于48000的城市'''
print(data['城市'][data['房价']>48000])
#其中 data['房价']>48000 的输出为：
# 0     True
# 1     True
# 2    False
# Name: 房价, dtype: bool
```
##### reindex,drop,apply函数的运用
```python
'''reindex(从frame指定的行索引中获取数据，如果没有则填充nan)'''
frame = pd.DataFrame(np.arange(9).reshape((3, 3)),index=['a', 'c', 'd'],columns=['Ohio', 'Texas', 'California'])
#    Ohio  Texas  California
# a     0      1           2
# c     3      4           5
# d     6      7           8
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
#输出：
#    Ohio  Texas  California
# a   0.0    1.0         2.0
# b   NaN    NaN         NaN
# c   3.0    4.0         5.0
# d   6.0    7.0         8.0
frame3 = pd.DataFrame(frame, index=['a', 'b', 'c', 'd'])#与上式frame2的reindex表达的同一个意思，输出也相同

'''drop【删除行axis=0与列axis=1，axis默认为0（与在Series中axis对应行列相同）】'''
frame3 = pd.DataFrame(frame, index=['a', 'b', 'c', 'd'])

frame4 = frame3.drop(['a', 'b'], axis=0)#删除行 
frame5=frame3.drop('California', axis=1)#删除列
print(frame4)
print(frame5)

'''apply应用函数(axis=1时是对行进行操作，axis=0时对列进行操作，这与drop中的使用正好相反)'''
frame = pd.DataFrame(np.arange(9).reshape((3, 3)),index=['a', 'c', 'd'],columns=['Ohio', 'Texas', 'California'])
#    Ohio  Texas  California
# a     0      1           2
# c     3      4           5
# d     6      7           8
def func(x):
    print(x)
frame.apply(func,axis=1)
#输出(使用apply,将每一行都遍历一遍)：
# Ohio          0
# Texas         1
# California    2
# Name: a, dtype: int32
# Ohio          3
# Texas         4
# California    5
# Name: c, dtype: int32
# Ohio          6
# Texas         7
# California    8
# Name: d, dtype: int32

def func(x):  #让小于3的数都等于0
    res = []
    for i in x:
        if i < 3:
            i = 0
        res.append(i)
    # print(x)
    return pd.Series(res, index=x.index)
result=frame.apply(func,axis=1) #每次调用func都对frame的单个行进行操作，返回一个Series,再通过apply去遍历所有行从而对所有行进行操作。最终每一行的Series共同形成新的DataFrame。
print(result)
#输出
#    Ohio  Texas  California
# a     0      0           0
# c     3      4           5
# d     6      7           8

result = frame.apply(lambda x: sum(x), axis=1)
result = frame.apply(lambda x: None, axis=1)
result = frame.apply(lambda x: pd.Series(x, index=['Ohio', 'Texas']), axis=1)

# 将指定操作(functional)应用到每一行/每一列
# functional可以使用匿名函数处理
# functional的返回值不同，处理结果不同
# 如果没有返回值 将会获取到全是None的Series
# 如果返回值是单个数值，result会是一个Series
# 如果返回值是一个Series,那result就是一个DataFrame
```
###### apply的课堂练习
```python
data = {'state': ['American:Ohio', 'American:Ohio', 'American:Ohio', 'China:BeiJing', 'China:BeiJing', 'China:BeiJing'],'year': ['2000', '2001', '2002', '2001', '2002', '2003'],'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]} 
frame = pd.DataFrame(data, index=['a', 'b', 'c', 'd', 'e', 'f'], columns=['year', 'pop', 'state'])
# 1.将year列的数据处理成数值型
frame.year = frame['year'].apply(int)
print(frame['year'].dtype)
# 2.提取state列中的国家,并作为frame新的列添加进来，列名为country
frame['country'] = frame['state'].apply(lambda x:x.split(':')[0])
frame['state'] = frame['state'].apply(lambda x:x.split(':')[1])
print(frame)
```


##### DataFrame的运算(与Series的运算相似)
```python
'''索引对齐'''
frame1 = pd.DataFrame(np.arange(9).reshape((3, 3)),index=['a', 'c', 'd'],columns=['Ohio', 'Texas','California'])
#    Ohio  Texas  California
# a     0      1           2
# c     3      4           5
# d     6      7           8
frame2 = pd.DataFrame(np.arange(9, 18).reshape((3, 3)),index=['a', 'c', 'd'],columns=['Ohio', 'Texas','California'])
print(frame1+frame2)#输出就是frame1和frame2对应元素相加

'''索引不对齐的情况'''
frame3 = pd.DataFrame(np.arange(9, 15).reshape((2, 3)),index=['a', 'c'],columns=['Ohio', 'Texas', 'California']) # 行索引不对齐
#    Ohio  Texas  California
# a     9     10          11
# c    12     13          14
frame4 = pd.DataFrame(np.arange(9, 15).reshape((3, 2)),index=['a', 'c', 'd'],columns=['Ohio', 'Texas']) # 列索引不对齐
#    Ohio  Texas
# a     9     10
# c    11     12
# d    13     14
print(frame1+frame3) 
#输出(frame3中没有'd'行，所以相加都为NaN)：
#    Ohio  Texas  California
# a   9.0   11.0        13.0
# c  15.0   17.0        19.0
# d   NaN    NaN         NaN  （NaN为浮点型，所以结果都为浮点型）
print(frame1+frame4)
#输出(frame4中没有'California列，所以相加都为NaN'):
#    California  Ohio  Texas
# a         NaN     9     11
# c         NaN    14     16
# d         NaN    19     21
```
##### DataFrame的其它操作
```python
print(frame1.head(2))# 查看前2行,默认查看前5行
print(frame1.tail(2))# 查看后面2行，默认值为5
print(frame1.describe())# 查看数值型的统计信息 (mean，std...)
frame5 = frame1+frame4
#    California  Ohio  Texas
# a         NaN     9     11
# c         NaN    14     16
# d         NaN    19     21
frame5['Ohio']['c'] = np.NaN
#    California  Ohio  Texas
# a         NaN   9.0     11
# c         NaN   NaN     16
# d         NaN  19.0     21
print(frame5.dropna(axis=1))#消除NaN所在的行(axis=0)或列(axis=1)
print(frame5.fillna(0))#把frame5中的NaN都使用0进行填充
```



### 待总结
```python
#.value_counts()
#"argmax() / argmin()返回的是序列中最大/小值的int位置,idxmax()返回的是最大值的行标签。"

#dataframe.index和dataframe.value在画折线图时的应用
```