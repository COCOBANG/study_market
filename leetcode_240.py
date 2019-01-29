# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:43:03 2019
leetcode_240.py
@author: yuxun
"""
"""
leetcode_4: searchMatrix
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.

Example:
Consider the following matrix:
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
Given target = 5, return true.
Given target = 20, return false.
"""

import numpy as np
import time

class Solution:
    def searchMatrix1(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if len(matrix) == 0:  #特殊情况处理
            return False
        elif len(matrix) > 0 and len(matrix[0]) == 0:
            return False
        
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            low_n, high_n = 0, n-1
            while low_n <= high_n:
                mid_n = (low_n+high_n) // 2
                if matrix[i][mid_n] < target:
                    low_n = mid_n + 1
                elif matrix[i][mid_n] > target:
                    high_n = mid_n - 1
                else:
                    return True
        return False
        
            
    def searchMatrix2(self, matrix, target):
        i = len(matrix) - 1
        j = 0
        while i >= 0 and j < len(matrix[0]):
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                i -= 1
            else:
                j += 1
        return False
  
    
    def searchMatrix3(self, matrix, target):
        # 基线条件(空矩阵)
        if len(matrix) == 0:  # 空矩阵
            return False
        elif len(matrix) > 0 and len(matrix[0]) == 0:
            return False
        # 先在对角线上找到符合条件matrix[i][i]<target<matrix[i+1][i+1]的点
        m, n  = len(matrix), len(matrix[0])
        min_mn = min(m, n)
        inf = float('inf')
        low_mn, high_mn = 0, min_mn-1
        while low_mn <= high_mn:
            mid_mn = (low_mn+high_mn) // 2
            # 为diag序列的左边添加一个-inf，右边添加一个inf的操作(这样总是能找到这样的元素--有可能取边界值)
            ### 这样处理的好处：(有序取两值的情况)
            ### (1) 不需要单独讨论边界情况，有统一的处理公式
            ### (2) 不需要关注越界、元素个数为1等问题
            if mid_mn == 0:  # 左边界取值
                if high_mn == 0:  # high_mn等于0时，向左添加一个-inf
                    lft, rgt = -inf, matrix[0][0]
                else:             # high_mn等于1时，取[0][0]和[1][1]
                    lft, rgt = matrix[0][0], matrix[1][1]
            elif mid_mn == min_mn-1:  # 右边界取值
                lft, rgt = matrix[mid_mn][mid_mn], inf
            else:  # 一般情况取值
                lft, rgt = matrix[mid_mn][mid_mn], matrix[mid_mn+1][mid_mn+1] 
            if rgt < target:
                low_mn = mid_mn + 1
            elif lft > target:
                high_mn = mid_mn - 1
            elif lft < target < rgt:  #找到符合条件的点就跳出循环
                break
            else:  # mid_mn和mid_mn+1和target相等时
                return True
        # 递归条件(二维情况:把矩阵分成两块来讨论)
        try:
            sub_matrix_1 = [i[:mid_mn+1] for i in matrix[mid_mn+1:]]
        except Exception as e:
            sub_matrix_1 = [[]]
        try:
            sub_matrix_2 = [i[mid_mn+1:] for i in matrix[:mid_mn+1]]
        except Exception as e:
            sub_matrix_2 = [[]]
        sM1 = self.searchMatrix2(sub_matrix_1, target)
        sM2 = self.searchMatrix2(sub_matrix_2, target)
        return (sM1 or sM2)

if __name__ == '__main__':
    solu = Solution()
    # test1
    target = 20
    matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
    print('test of searchMatrix:')
    print('matrix is:')
    print(np.array(matrix))
    print(f'target is: {target}')
    start = time.time()
    print(f'searchMatrix1 output is: {solu.searchMatrix1(matrix, target)}')
    print(f'searchMatrix1 running time is: {round(time.time()-start, 8)}')
    start = time.time()
    print(f'searchMatrix2 output is: {solu.searchMatrix2(matrix, target)}')
    print(f'searchMatrix2 running time is: {round(time.time()-start, 8)}')
    start = time.time()
    print(f'searchMatrix3 output is: {solu.searchMatrix3(matrix, target)}')
    print(f'searchMatrix3 running time is: {round(time.time()-start, 8)}')
    
    # test2
    target = 131540
    m, n = 5000, 5500
    matrix = np.zeros((m, n), dtype='int')
    matrix[0][0] = np.random.randint(0, 20)
    for i in range(m):
        for j in range(n):
            if i == 0 and j > 0:
                matrix[i][j] = matrix[i][j-1] + np.random.randint(0, 20)
            elif i > 0 and j == 0:
                matrix[i][j] = matrix[i-1][j] + np.random.randint(0, 20)
            else:
                matrix[i][j] = max(matrix[i][j-1], matrix[i-1][j]) + np.random.randint(0, 20)
    matrix = [list(matrix[i]) for i in range(m)]
    print('test of searchMatrix:')
    print('matrix is:')
    print(np.array(matrix))
    print(f'target is: {target}')
    start = time.time()
    result = solu.searchMatrix1(matrix, target)
    end = time.time()
    print(f'searchMatrix1 output is: {result}')
    print(f'searchMatrix1 running time is: {round(end-start, 8)}')
    start = time.time()
    result = solu.searchMatrix2(matrix, target)
    end = time.time()
    print(f'searchMatrix2 output is: {result}')
    print(f'searchMatrix2 running time is: {round(end-start, 8)}')
    start = time.time()
    result = solu.searchMatrix3(matrix, target)
    end = time.time()
    print(f'searchMatrix3 output is: {result}')
    print(f'searchMatrix3 running time is: {round(end-start, 8)}')
