# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 15:56:08 2019
div_2_class.py
@author: yuxun
"""

import numpy as np
import math

def div_2_class(list_input):
    """
    将一个输入的list或np.array分成两堆，使得两堆与之和的差最小的函数。注意，输入的list或np.array的内容必须是正整数。
    另外，分堆的解可以不止一个。
    Inputs:
        list_input: 包含正整数的list或np.array，个数不限
    Outputs:
        class1: 第一堆数(注意：是一个n*m维数组，n为分堆解的个数，m为输入数组的数字个数；选择出来的显示，否则显示为0)
        class2: 第二堆数
        delta: class1和class2的差
    (**)解题思路：
    (1) 设输入序列n的各元素之和为sum，“两堆之和的差最小”，可以转换为“选择一些数字组成其中的一堆，其和在满足sum(1)<=floor(sum/2)条件下，
    使得sum(1)最大”，显然此时选择出来的这一堆数之和sum(1)和剩下的数之和sum(2)的差就是最小的。(换句话说，就是让两堆数之和尽量接近sum/2)；
    (2) 前述转换的描述便是一个“0-1 背包问题”，可以利用动态规划求解：
        (2.1) 定义一个价值矩阵value，第i行讨论的是第i个元素，第j列讨论的是背包容量为j，value[i, j]就是背包容量为j时，
              考虑完第i个元素是否添加后的价值(最大化的对象，这里即为和sum(1))；
        (2.2) value的递推公式：value[i, j] = max(not_take, take) = max(value[i-1, j], value[i-1, j-n(i)] + n[i])，
              其中n(i)为第i个元素的价值，这也是一个独立的子问题的解
              (**理解**)假设前面(i-1)个元素的最大化问题都已经解决了(不同的容量j)，那么容量j时考虑了第i个元素后的价值计算，只需要看两种情况：
              (a) 不选择i，此时的价值(sum(1))就和(i-1)个元素在j容量约束时的最大价值是一样的；
              (b) 选择i，那么此时的价值(sum(1))就是(i-1)个元素在(j-n[i])容量约束时的最大价值，加上i元素的价值n[i]；
              只需要在(a)和(b)里选择最大的价值，在value[i, j]这个节点上就做到了最大化，因而有(2.2)式；
        (2.3) 边界条件：
              背包容量为0时，value[:, 0] = 0；
              考虑第0个元素时，value[0, :] = 0；
              递推时，如果j<n(i)，让take取很小的值(如-1，一定是不取的情况，这意味着容量j还没有元素i的价值大)
        (2.4) value[:,-1]这一列的最大值，即为sum(1)的最大值
    (3) (2)解决了选出来的堆的和的最大值是多少的问题，但是还需要知道具体取了哪些数，这里如下处理：
        (3.1) 假设用二进制为序列n进行编码，总共有len(n)位，假如选择了第i个元素n[i]，则第i位为1，否则为0，比如：
              n包含了5个正整数，选择了其中的第2、5个数，则生成一个编码01001，
              利用这个编码即可从序列n中取数了。
        (3.2) 取数编码is_take也有递推公式(思路上和value的递推是一致的)：
                is_take[i, j] = is_take[i, j-1], 如果不取第i个元素
                is_take[i, j] = 2^(num-i) + is_take[i-1, j-n[i]], 如果取第i个元素
    """
    # (1)获得参数和初始化数组
    sup = math.floor(sum(list_input)/2)  # 背包约束的最大值
    num = len(list_input)  # 输入数组的数字个数
    value = np.zeros((num+1, sup+1))  # value和is_take的大小都是[(num+1)*(sup+1)]的(因为有前述边界条件(2.3))
    is_take = np.zeros((num+1, sup+1), dtype=int)
    # (2)利用递推公式进行循环遍历，获得整个value和is_take矩阵的取值
    for i in range(1, num+1):
        num_to_take_or_not = list_input[i-1]  # 第i个元素(由于value/is_take第0行表示不拿数，因此这里取数下标是(i-1))
        for j in range(1, sup+1):             # 查看第i个元素在不同的容量约束j([1, sup])下的取值情况
            # value[i, j]
            not_take = value[i-1, j]   # not_take: 不取第i个元素的价值
            if j>=num_to_take_or_not:  # take: 取第i个元素的价值，注意j>=n(i)的约束
                take = value[i-1, j-num_to_take_or_not] + num_to_take_or_not
            else:
                take = -1
            value[i, j] = max(not_take, take)
            # is_take[i, j]
            if take >= not_take:
                is_take[i, j] = 2**(num-i) + is_take[i-1, j-num_to_take_or_not]
            else:
                is_take[i, j] = is_take[i-1, j]
    # (3)通过最后一列value[:,-1]求解
    ## (3.1)获得value最大时的取数编码
    sup_value = value[:, -1]
    sup_value_max = max(sup_value)
    sup_value_max_is_take = is_take[:, -1][np.where(sup_value==sup_value_max)]  # 取最大价值时的取数编码
    sup_value_max_is_take = np.unique(sup_value_max_is_take)  # 去重，需要注意的是，取数编码可能不止一个
    length_max_is_take = len(sup_value_max_is_take)
    # (3.2)依据取数编码，从list_input里面取数
    pos_info = np.array([list(bin(i)[2:].zfill(num)) for i in sup_value_max_is_take])  # 将每一个取数编码转换为二进制，得到一个位置信息矩阵pos_info
    class1 = np.stack([list_input for n in range(length_max_is_take)], axis=0) * (pos_info=='1')  # 获得class1和class2，注意在numpy里矩阵直接相乘是"按位相乘"
    class2 = np.stack([list_input for n in range(length_max_is_take)], axis=0) * (pos_info=='0')
    delta = abs(class1.sum(axis=1) - class2.sum(axis=1))  # 求两堆数之和的差

    return class1, class2, delta


if __name__ == '__main__':
    # test1
    list_input = np.array([1, 2, 6, 9])
    class1, class2, delta = div_2_class(list_input)
    print('input:')
    print(list_input)
    print('class1:')
    print(class1)
    print('class2:')
    print(class2)
    print('delta:')
    print(delta)
    
    # test2
    list_input = [1, 2, 6, 9, 5, 19, 14, 1, 8]
    class1, class2, delta = div_2_class(list_input)
    print('input:')
    print(list_input)
    print('class1:')
    print(class1)
    print('class2:')
    print(class2)
    print('delta:')
    print(delta)
