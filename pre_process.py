# -*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import mode
import common
import myanalysis
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MultiLabelBinarizer
HEADER = myanalysis.HEADER

GROUP_NUM = myanalysis.GROUP_NUM # 对区间进行分组的组数
COL = 'I1' #测试固定列
TOP_K = 9 # 在绘图显示时只显示前TOP_K个
# traindata = pd.read_table('./data/train.small.csv')
traindata = pd.read_table('./data/train.small.csv')
# print(traindata.mode().ix[0])

a = pd.DataFrame({'one':[1, 2, 1, 2, np.nan], 'two':[np.nan, 3, 4, 3, 7], 'three':[np.nan, 0, 3, np.nan, 0]})
# a.fillna(mode(a)[0])
b = a.mode().ix[0]
mode_dict = dict(zip(a.columns, b))
c = a.fillna(b)
d = a.fillna(mode_dict)




testdata = pd.DataFrame({'pet': ['cat', 'dog', 'dog', 'fish'],
                         'age': [4 , 6, 3, 3],
                         'salary':[4, 5, 1, 1]})
a = LabelEncoder().fit_transform(testdata['pet'])
# print(a)
# print(OneHotEncoder( sparse=False ).fit_transform(a.reshape(-1,1)))
#LabelBinarizer().fit_transform(testdata['pet'])
mlb = MultiLabelBinarizer()
# b = mlb.fit_transform([(1, 5, 9), (2, 10, 11), (3, 6, 8), (2, 4, 7)])
# print(b)
# print(b.shape)
# print(mlb.classes_)





# 测试所有特征与其频数
def test_feature_count(data):
    data = myanalysis.fill_with_mode(data)
    d = myanalysis.feature_count(data, HEADER, GROUP_NUM)
    # 经过这个过程，已经将连续特征域进行了填充，现在data数据是没有缺失值的，而且全部是离散的
    cols = data.columns.drop('Label')
    data_vec = pd.get_dummies(data[cols], columns=cols, prefix=cols)

    # 跑模型
    data_vec['intercept'] = 1.0
    logit = sm.Logit(data['Label'], data_vec)
    result = logit.fit()


    # print(data_vec.shape)
    # print(data_vec.ix[:, :20])
    # x = d.index[:13]
    # y = d.values[:13]
    # plt.figure(figsize=(8,6))
    # plt.ylabel('Feature')
    # plt.xlabel('counts')
    # plt.yticks(np.arange(x.shape[0]), x, rotation=0)
    # plt.barh(np.arange(x.shape[0]), y)
    # plt.subplots_adjust(left=0.2)
    # # plt.savefig('./data/feature_count.png', dpi=100)
    # plt.show()
test_feature_count(traindata)




# 一旦整个数据集所有的特征的频数统计出来了，这个函数就没有用了，因为连续特征离散化之后有很多区间没有值，在ont-hot
# 过程中会被丢弃，最红的特征会比理论上要少一些
def test_feature_fre(data):
    col_dict = myanalysis.feature_fre(data, HEADER, GROUP_NUM)
    print(len(col_dict))
# test_feature_fre(traindata)

def test_integer_col_dict(data):
    d = myanalysis.integer_col_dict(data[COL], GROUP_NUM, 10)
    print(d)
# test_integer_col_dict(traindata)
def test_categorical_col_dict(data):
    d = myanalysis.categorical_col_dict(data[COL], 130)
    print(d)
    print(len(d))
# test_categorical_col_dict(traindata)

# 测试每一种特征的空值与非空值情况
def test_check_nan(data):
    df = myanalysis.check_nan(data)
    print(df)
    df.plot(kind='bar', stacked=True)
    plt.show()
# test_check_nan(traindata)
# 结果显示连续特征中控制大约占到30%

# 测试每一种特征的分布情况
def test_all_distribution(data):
    cols = ['I1', 'I2', "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13"]
    # cols = [COL]
    arr = myanalysis.integer_distribution(traindata, cols)
    for col, values in arr.iteritems():
        plt.figure(figsize=(8,6), dpi=100)
        plt.title("Feature " + col)
        figurepath = './data/Feature_' + col + '.png'
        plt.xlabel("interval")
        plt.ylabel("count")
        value = values[0][:TOP_K]
        xtick = ['[' + str(values[1][i]) + ',' + str(values[1][i+1]) + ']' for i in range(TOP_K)]
        plt.bar(np.arange(len(value)), value)
        plt.xticks(np.arange(len(value)), xtick, rotation=45)
        # plt.savefig(figurepath, dpi=100)  # 图保存在data文件夹中,使用的GROUP_NUM = 15
        # plt.subplots_adjust(bottom=0.2)
        plt.show()
# test_all_distribution(traindata)

# 查看每一种连续特征的众数特征
def test_mode(data):
    cols = HEADER[1:14]
    mode_arr = [mode(data[col]) for col in cols]
    print(mode_arr)
# test_mode(traindata)
# 结果显示大部分的连续特征的众数为0或者1

def test_categorical_distribution(data):
    cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"]
    for ele in myanalysis.categorical_distribution(data, cols):
        ele = ele[:12]
        plt.figure(figsize=(8,6), dpi=100)
        plt.title("Feature " + ele.name)
        plt.xlabel('value')
        plt.ylabel('counts')
        plt.bar(np.arange(ele.values.shape[0]), ele.values)
        plt.xticks(np.arange(ele.values.shape[0]), ele.index, rotation='vertical')
        figurepath = './data/Feature_'+ele.name+'.png'
        # plt.savefig(figurepath, dpi=100)  # 图保存在data文件夹中
        # plt.subplots_adjust(bottom=0.2)
        # plt.show()
# test_categorical_distribution(traindata)
#结果显示和连续特征一样，取值较为集中，分布非常不均匀


# 产生test.small.csv文件
def generate_test_small_csv():
    f = open('./data/train.txt')
    f_test = open('./data/test.small.csv', 'w')
    f_test.write('\t'.join(HEADER) + '\n')
    index = 1
    while True:
        if index > 2000:
            break
        if np.random.random() < 0.2:
            f_test.write(f.readline())
            index = index + 1
    f.close()
    f_test.close()
# generate_test_small_csv()


