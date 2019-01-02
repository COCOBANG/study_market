# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:49:07 2019
longest_common_subsequence.py
@author: yuxun
"""

import numpy as np


# LCS
def LCS(list_a, list_b):
    """
    计算两个list之间的最长公共子序列(Longest Common Subsequence)，指的是求解两个list的元素中“有顺序的共同部分”，如：
    list_a = [1, 4, 5, 3, 7]; list_b = [3, 1, 5, 7, 9]
    首先，如果只讨论公共部分，那么就是它们的交集，即[1, 3, 5, 7]。但是元素'3'的顺序在两个list中是不一致的，按list_a中的顺序
    的交集可以写为[1, 5, 3, 7]，而按list_b中的顺序的交集可以写为[3, 1, 5, 7]，显然，所谓“有顺序的共同部分”就是[1, 5, 7]了，
    它在list_a和list_b中的顺序是一致的。另外，解可以不止一个。
    Inputs:
        list_a, list_b: 比较的序列，可以是字符串，也可以是list或np.array
    Outputs:
        list_LCS: 最长公共子序列，选择的部分非空(string)或非nan(number)，而未被选择的部分为空或nan
    (**)解题思路：
    (1) list_a的每个元素视为一行，list_b的每个元素视为一列，最大化的是“顺序的公共部分”的长度，仍然用动态规划的思路求解；
    (2) 定义价值函数value(这里就是(1)所述的长度了)，value[i, j]表示list_a从0看到i元素、list_b从0看到j元素为止的最大长度。
        (2.1) 有递推公式：
            value[i, j] = value[i-1, j-1] + 1, 如果list_a[i]==list_b[j]
            value[i, j] = max(value[i, j-1], value[i-1, j]), 如果list_a[i]!=list_b[j]
        (**理解**)仍然是假设前面问题都已经解决了(不同的i/j)，如果list_a第i个元素等于list_b第j个元素，那么显然最大长度就是value[i-1, j-1]+1，
        如果list_a第i个元素不等于list_b第j个元素，那就保持之前的最大值状态(注意这里是“保持”，意味着并不要求公共部分是连续的——
        要求连续的是最长公共子串问题)，而所谓之前，就是[i, j-1]和[i-1, j](只少看一个元素)，因此保持之前最大值就是求这两个节点的最大值。
        (2.2) 边界条件：
        考虑list_a的第0个元素时，value[0, :] = 0；
        考虑list_b的第0个元素时，value[:, 0] = 0
    (3) (2)解决了选出来的子序列长度的最大值是多少的问题，但是还需要知道具体取了哪些元素，这里如下处理：
    (3.1) 假设用二进制为序列list_a进行编码(由于选出的是公共部分，所以用list_a和list_b是无差别的)，总共有len(list_a)位，
          假如选择了list_a的第i个元素list_a[i]，则第i位为1，否则为0，比如：
          list_a包含了5个元素，选择了其中的第2、5个元素，则生成一个编码01001，利用这个编码即可从序列list_a中取数了。
    (3.2) 取数编码is_take也有递推公式(思路上和value的递推是一致的)：
            is_take[i, j] = 2^(num-i) + is_take[i-1, j-1], 如果list_a[i]==list_b[j]
            is_take[i, j] = is_take[i-1, j], 如果list_a[i]!=list_b[j]且value[i-1, j]>value[i, j-1]
            is_take[i, j] = is_take[i, j-1], 如果list_a[i]!=list_b[j]且value[i, j-1]>value[i-1, j]
    """
    # (1)获得参数和初始化数组
    length_a = len(list_a)  # list_a长度
    length_b = len(list_b)  # list_b长度
    value = np.zeros((length_a+1, length_b+1)) # value和is_take的大小都是[(len_a+1)*(len_b+1)]的(因为有前述边界条件(2.3))
    is_take = np.zeros((length_a+1, length_b+1), dtype=int)
    # (2)利用递推公式进行循环遍历，获得整个value和is_take矩阵的取值
    for i in range(1, length_a+1):
        for j in range(1, length_b+1):
            # value[i, j]: 价值信息
            if list_a[i-1] == list_b[j-1]:
                value[i][j] = value[i-1][j-1] + 1
            else:
                value[i][j] = max(value[i-1][j], value[i][j-1])
            #is_take[i, j]: 取数信息
            if list_a[i-1] == list_b[j-1]:            
                is_take[i][j] = 2**(length_a-i) + is_take[i-1][j-1]
            else:             
                if value[i-1][j] > value[i][j-1]:  # 注意，这里无论取不取等号都有可能出现bug，因为相等时理论上应该要把两个is_take的信息都包含进来
                    is_take[i][j] = is_take[i-1][j]
                else:
                    is_take[i][j] = is_take[i][j-1]
    # (3)通过value和is_take求解
    ## (3.1)获得value最大时的取数编码
    max_value = value.max()
    max_is_take = is_take[np.where(value==max_value)]  # 取最大价值时的取数编码
    max_is_take = np.unique(max_is_take)  # 去重，需要注意的是，取数编码可能不止一个
    length_max_is_take = len(max_is_take)
    # (3.2)依据取数编码，从list_input里面取数
    pos_info = np.array([list(bin(i)[2:].zfill(length_a)) for i in max_is_take])  # 将每一个取数编码转换为二进制，得到一个位置信息矩阵pos_info
    if isinstance(list_a, str):  # 分情况讨论，因为输入的list可能是string也可能是list或np.array
        list_LCS = np.stack([np.array(list(list_a)) for n in range(length_max_is_take)], axis=0)
        list_LCS[pos_info=='0'] = ''
    elif isinstance(list_a, list):
        list_LCS = np.stack([np.array(list_a).astype('float') for n in range(length_max_is_take)], axis=0)
        list_LCS[pos_info=='0'] = np.nan
    else:
        list_LCS = np.stack([list_a.astype('float') for n in range(length_max_is_take)], axis=0)
        list_LCS[pos_info=='0'] = np.nan
        
    return list_LCS


# LCString
def LCString(list_a, list_b):
    """
    计算两个list之间的最长公共子串(Longest Common Substring)，指的是求解两个list的元素中“连续的有顺序的共同部分”，定义上，
    该问题只是比LCS问题多了一个“连续”条件，因此递推公式会不一样。另外，同样地，解可以不止一个。
    Inputs:
        list_a, list_b: 比较的序列，可以是字符串，也可以是list或np.array
    Outputs:
        list_LCString: 最长公共子串，选择的部分非空(string)或非nan(number)，而未被选择的部分为空或nan
    (**)解题思路：只简单说明和LCS的异同
    (2) value有递推公式：
            value[i, j] = value[i-1, j-1] + 1, 如果list_a[i]==list_b[j]
            value[i, j] = 0, 如果list_a[i]!=list_b[j]
        (**理解**)仍然是假设前面问题都已经解决了(不同的i/j)，如果list_a第i个元素等于list_b第j个元素，那么显然最大长度就是value[i-1, j-1]+1，
        如果list_a第i个元素不等于list_b第j个元素，由于有“连续”条件约束，因此此时value[i, j]为0(意味着重新开始看最大长度)。
        (2.2) 边界条件：
        考虑list_a的第0个元素时，value[0, :] = 0；
        考虑list_b的第0个元素时，value[:, 0] = 0
    (3) 取数编码is_take也有递推公式(思路上和value的递推是一致的)：
            is_take[i, j] = 2^(num-i) + is_take[i-1, j-1], 如果list_a[i]==list_b[j]
            is_take[i, j] = 0
    """
    # (1)获得参数和初始化数组
    length_a = len(list_a)  # list_a长度
    length_b = len(list_b)  # list_b长度
    value = np.zeros((length_a+1, length_b+1)) # value和is_take的大小都是[(len_a+1)*(len_b+1)]的(因为有前述边界条件(2.3))
    is_take = np.zeros((length_a+1, length_b+1), dtype=int)
    # (2)利用递推公式进行循环遍历，获得整个value和is_take矩阵的取值
    for i in range(1, length_a+1):
        for j in range(1, length_b+1):
            # value[i, j]: 价值信息
            if list_a[i-1] == list_b[j-1]:
                value[i][j] = value[i-1][j-1] + 1
            else:
                value[i][j] = 0
            #is_take[i, j]: 取数信息
            if list_a[i-1] == list_b[j-1]:            
                is_take[i][j] = 2**(length_a-i) + is_take[i-1][j-1]
            else:             
                is_take[i][j] = 0
    # (3)通过value和is_take求解
    ## (3.1)获得value最大时的取数编码
    max_value = value.max()
    max_is_take = is_take[np.where(value==max_value)]  # 取最大价值时的取数编码
    max_is_take = np.unique(max_is_take)  # 去重，需要注意的是，取数编码可能不止一个
    length_max_is_take = len(max_is_take)
    # (3.2)依据取数编码，从list_input里面取数
    pos_info = np.array([list(bin(i)[2:].zfill(length_a)) for i in max_is_take])  # 将每一个取数编码转换为二进制，得到一个位置信息矩阵pos_info
    if isinstance(list_a, str):  # 分情况讨论，因为输入的list可能是string也可能是list或np.array
        list_LCString = np.stack([np.array(list(list_a)) for n in range(length_max_is_take)], axis=0)
        list_LCString[pos_info=='0'] = ''
    elif isinstance(list_a, list):
        list_LCString = np.stack([np.array(list_a).astype('float') for n in range(length_max_is_take)], axis=0)
        list_LCString[pos_info=='0'] = np.nan
    else:
        list_LCString = np.stack([list_a.astype('float') for n in range(length_max_is_take)], axis=0)
        list_LCString[pos_info=='0'] = np.nan
    
    return list_LCString


#LIS
def LIS(list_input):
    """
    求一个输入list的最长递增子序列(Longest Increased Subsequence)，所谓最大递增子序列，是指：
    设list_input=[a1, a2, …, an]是n个元素的序列，它的递增子序列是这样一个子序列IS=[a[k1=, a[k2], …, a[km]]，
    其中k1 <k2 <… <km且a[K1] <a[k2] <… <a[km]，而最大递增子序列就是这些IS中长度最长的那个。
    Inputs:
        list_input: 输入的序列，可以是字符串，也可以是list或np.array
    Outputs:
        list_LIS: 最长递增子序列，选择的部分非空(string)或非nan(number)，而未被选择的部分为空或nan
    (**)解题思路：可以通过求解list_input和list_input_sorted的最大公共子序列(LCS)得到
    """
    list_input_sorted = list_input.copy()
    list_input_sorted.sort()
    list_LIS = LCS(list_input, list_input_sorted)
    return list_LIS




if __name__ == '__main__':
    # test1 of LCS
    list_a = 'fish'
    list_b = 'fosh'
    print('test of LCS:')
    print('input:')
    print(f'list_a: {list_a}')
    print(f'list_b: {list_b}')
    print('LCS of them:')
    print(LCS(list_a, list_b))
    # test1 of LCS
    list_a = [1, 4, 5, 3, 7]
    list_b = [3, 1, 5, 7, 9]
    print('test of LCS:')
    print('input:')
    print(f'list_a: {list_a}')
    print(f'list_b: {list_b}')
    print('LCS of them:')
    print(LCS(list_a, list_b))
    # test of LCString
    list_a = 'HISH'
    list_b = 'VISTA'
    print('test of LCString:')
    print('input:')
    print(f'list_a: {list_a}')
    print(f'list_b: {list_b}')
    print('LCString of them:')
    print(LCString(list_a, list_b))
    # test of LIS
    list_input = np.random.randint(100, size=10)
    print('test of LIS:')
    print(f'list_input: {list_input}')
    print('LIS of list_input:')
    print(LIS(list_input))