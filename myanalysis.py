# -*- encoding:utf-8 -*-
import pandas as pd
import common
import math

GROUP_NUM = 50
HEADER=["Label", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"]


def ont_hot_dataset(data, cols, n):
    col_dict = feature_fre(data, cols, n)

# 统计特征与特征频率（包括连续特征与离散特征）
# 返回一个Series，index为特征名，value为特征出现的频数
def feature_count(data, cols, n):
    f_count = pd.Series()  #先处理连续特征 每个值减掉特征域均值然后除以区间长度得到属于哪一区间的区间编号，
    for col in cols[1:14]:  # 这里每一个区间离散化为了一个特征
        min_val = data[col].min(); max_val = data[col].max();step = (max_val - min_val) / float(GROUP_NUM)
        data[col] = data[col].map(lambda x: str(int((x - min_val) / step)))  # 这里用了str，因为后面的字符串连接需要
        tmp = data[col].value_counts()
        tmp.index = col + '_' + tmp.index  # 将特征域名誉区间编号链接形成新的特征名
        f_count = f_count.append(tmp)
    for col in cols[14:]:  #再处理离散特征
        tmp = data[col].value_counts()  # 类别特征直接用特征域名加上特征取值形成新的特征
        tmp.index = col + '_' + tmp.index
        f_count = f_count.append(tmp)
    return f_count


# 将连续特征离散成不同的区间，并对特征进行编号，按照（特征名_区间号：编号）的键值对方式进行组合
# 这里的编号为从startindex开始的顺序编号
def integer_col_dict(x, n, startindex):
    return {x.name+ '_' + str(i): startindex + i for i in range(n)}
# 将类别特征进行编号，按照（特征名_特征值：编号）的键值对方式进行编号，编号也是从startindex开始的
def categorical_col_dict(x, startindex):
    return {x.name + '_' + str(name): startindex + i for i, name in enumerate(set(x))}

# 合并连续特征和类别特征形成的所有的键值对（特征：编号）的映射
def feature_fre(data, cols, n):
    d = {}  # 先处理连续特征，有13种连续特征，一共形成13*n种离散化的特征
    for ele in [integer_col_dict(data[col], n, index * n) for index, col in enumerate(cols[1:14])]:
        d.update(ele)
    startindex = 130
    for col in cols[14:]:
        tmp_dict = categorical_col_dict(data[col], startindex)
        d.update(tmp_dict)
        startindex = startindex + len(tmp_dict)
    return d

# 找到每一种特征的空值和非空值的情况，返回一个DataFrame，index为特征名，columns有两项：空值数量和非空值数量
def check_nan(data):
    d = {}  # key为列名，value为一个dict，里面包含空值数量很非空值数量
    for col in data.columns:
        tmp = sum(data[col].isnull())
        if tmp > 0:
            d[col] = dict(null_value=tmp, notnull_value=data.shape[0] - tmp)
    return pd.DataFrame(d).T

# 找到指定特征的分布情况（针对连续特征属性）
# 返回一个字典，字典key为特征列元素为tuple，每一个tuple有两个元素，第一为特征的区间分布数组，第二为区间坐标数组
def integer_distribution(data, cols):
    return {col: (common.get_distribution(data[col][data[col].notnull()], GROUP_NUM)) for col in cols}

# 返回所有离散特征分布情况（列表），每一个元素有两项，第一是特征名，第二是特征出现次数
def categorical_distribution(data, cols):
    return [data[col].value_counts() for col in cols]

#  缺失值处理：将读取的原始数据进行众数填充，就地修改（节省内存）
def fill_with_mode(data):
    mode_df = data.mode()
    mode_df = mode_df.drop('Label', axis=1)
    return data.fillna(mode_df.ix[0], inplace=True)

############################################
# df = pd.DataFrame([
#     ['green', 'M', 10.1, 'class1'],
#     ['red', 'L', 13.5, 'class2'],
#     ['blue', 'XL', 15.3, 'class1']])
# df.columns = ['color', 'size', 'prize', 'class label']
# size_mapping = {
#     'XL': 3,
#     'L': 2,
#     'M': 1}
# df['size'] = df['size'].map(size_mapping)
# class_mapping = {label:idx for idx,label in enumerate(set(df['class label']))}
# df['class label'] = df['class label'].map(class_mapping)
# col = df.columns.drop(['class label', 'prize'])
# print(pd.get_dummies(df, columns=col, drop_first=True))
#get_dummies函数在pandas0.19.2当前版本中有着这样的特性，
#如果使用pd.get_dummies(df[col)函数只会把特征至为字符的列进行dummy化，其他数值列（即便是分类特征）也不会dummy化
#必须添加columns参数进行指定
#skleran中的OneHotEncoder只能对数值型特征进行one-hot，字符型特征不行，OneHotEncoder还有个fit_transform函数
#可以直接对记录进行one-hot，例如OneHotEncoder(sparse = False).fit_transform( testdata[['age']] )
#OneHotEncoder的输入必须是二维,OneHotEncoder有一个好处就是在训练集中fit之后可以在测试集中transform再次使用
