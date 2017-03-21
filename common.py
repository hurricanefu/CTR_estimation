# -*- encoding:utf-8 -*-
import numpy as np

# 获取数组x的分布，返回两个数组，第一个是n个区间上的数量数组
# 第二个是区间坐标数组（间隔为固定值）
def get_distribution(x, n):
    step = (x.max() - x.min()) / n
    sumarr = [0 for i in range(n)]
    min_val = x.min()
    value_arr = [float("%.4f" % (min_val + i * step)) for i in range(n)]
    for ele in x:
        index = int((ele - min_val) / step)
        if index == n:
            sumarr[n-1] = sumarr[n-1] + 1
        else:
            sumarr[index] = sumarr[index] + 1
    # print(sumarr)
    # print(value_arr)
    return sumarr, value_arr
