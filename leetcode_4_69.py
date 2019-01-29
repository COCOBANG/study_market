# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 09:46:01 2019
leetcode_4_69.py    
@author: yuxun
"""

"""
leetcode_4: findMedianSortedArrays
There are two sorted arrays nums1 and nums2 of size m and n respectively.
Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
You may assume nums1 and nums2 cannot be both empty.

Example 1:
nums1 = [1, 3]
nums2 = [2]
The median is 2.0
Example 2:
nums1 = [1, 2]
nums2 = [3, 4]
The median is (2 + 3)/2 = 2.5
"""

"""
leetcode_69:
Implement int sqrt(int x).
Compute and return the square root of x, where x is guaranteed to be a non-negative integer.
Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.

Example 1:
Input: 4
Output: 2
Example 2:
Input: 8
Output: 2
"""

class Solution:
    
    # leetcode_4(1):
    def findMedianSortedArrays(self, nums1, nums2):
        # (1)调整输入数组，使得short、long分别表示较长的和较短的
        if len(nums1) <= len(nums2):
            short ,long = nums1, nums2
        else:
            short, long = nums2, nums1  
        len_sum = len(nums1) + len(nums2)
        inf = float('inf')
        # (2)如果short是空，那么中位数只需要关注long
        if len(short) == 0:
            if len(long)%2 == 0:
                median = (long[len(long)//2-1] + long[len(long)//2]) / 2
            else:
                median = long[len(long)//2]
            return median
        # (3)一般情况时，给short和long各添加一个下限和上限，进行二分查找
        short = [-inf] + short + [inf]
        long = [-inf] + long + [inf]
        low = 1
        high = len(short) - 1
        while low <= high:
            mid = (low+high) // 2
            i = mid                 # 实际上是在short数组里进行查找
            j = (len_sum+4)//2 - i  # 只要short数组位置给定了，long数组的位置就给定了
            if (short[i-1] <= long[j]) and (long[j-1] <= short[i]):  # 查找的条件
                if len_sum%2 == 0:  # 依据奇偶性给出中位数计算公式
                    median = (max(short[i-1], long[j-1]) + min(short[i], long[j])) / 2
                else:
                    median = min(short[i], long[j])
                return median
            elif short[i] < long[j-1]:  # short[i]的位置靠左了，调整low
                low = mid + 1
            elif short[i-1] > long[j]:  # short[i]额位置靠右了，调整high
                high = mid - 1
    
    # leetcode_4(2):
    def findMedianSortedArrays_inf(self, nums1, nums2):
        # (1)调整输入数组，使得short、long分别表示较长的和较短的
        if len(nums1) <= len(nums2):
            short ,long = nums1, nums2
        else:
            short, long = nums2, nums1
        len_sum = len(nums1) + len(nums2)
        inf = float('inf')
        # (2)如果short是空，那么中位数只需要关注long
        if len(short) == 0:
            if len(long)%2 == 0:
                median = (long[len(long)//2-1] + long[len(long)//2]) / 2
            else:
                median = long[len(long)//2]
            return median
        # (3)一般情况时，进行二分查找(与findMedianSortedArrays不同的是，这里的边界情况进行了分类讨论)
        low = 0
        high = len(short)
        while low<=high:
            # i赋值(short数组查找)
            mid = (low+high) // 2
            i = mid
            j = len_sum//2 - i
            # 边界情况赋值
            if 0 < i < len(short):
                short_left_last = short[i-1]
                short_right_first = short[i]
            elif i == 0:
                short_left_last = -inf
                short_right_first = short[0]
            elif i == len(short):
                short_left_last = short[-1]
                short_right_first = inf
            if 0 < j < len(long):
                long_left_last = long[j-1]
                long_right_first = long[j]
            elif j == 0:
                long_left_last = -inf
                long_right_first = long[0]
            elif j == len(long):
                long_left_last = long[-1]
                long_right_first = inf
            # 查看是否满足了查找条件，否则变更low、high
            if (short_left_last <= long_right_first) and (long_left_last <= short_right_first):
                if len_sum%2 == 0:  # 依据奇偶性给出中位数计算公式
                    median = (max(short_left_last, long_left_last) + min(short_right_first, long_right_first)) / 2
                else:
                    median = min(short_right_first, long_right_first)
                return median
            elif short_right_first < long_left_last:  # short[i]的位置靠左了，调整low
                low = mid + 1
            elif short_left_last >long_right_first:  # short[i]的位置靠右了，调整high
                high = mid - 1

    # leetcode_69:
    def mySqrt(self, x):
        """
        查找的值应该满足如下条件：
        mid**2 <= x < (mid+1)**2
        """
        # 特殊情况处理
        if x == 0: 
            return 0
        # 按条件进行二分查找
        low = 1
        high = x
        while low<=high:
            mid = (low+high) // 2
            if x >= (mid+1)**2:
                low = mid + 1
            elif x < mid**2:
                high = mid - 1
            else:
                return mid


if __name__=='__main__':
    
    solu = Solution()
    
    # test1 of findMedianSortedArrays:
    nums1 = [-1, 0]
    nums2 = [1, 5, 8, 10, 12, 14, 16, 18]
    print('test1 of findMedianSortedArrays:')
    print('Input:')
    print(f'nums1: {nums1}')
    print(f'nums2: {nums2}')
    print(f'Median is: {solu.findMedianSortedArrays(nums1, nums2)}')
    
    # test2 of findMedianSortedArrays:
    nums1 = [1, 2, 3, 7, 9]
    nums2 = [5, 8, 10, 12, 14]
    print('test2 of findMedianSortedArrays:')
    print('Input:')
    print(f'nums1: {nums1}')
    print(f'nums2: {nums2}')
    print(f'Median is: {solu.findMedianSortedArrays(nums1, nums2)}')
    
    # test3 of findMedianSortedArrays:
    nums1 = [1, 2, 3]
    nums2 = [4, 5, 6]
    print('test3 of findMedianSortedArrays:')
    print('Input:')
    print(f'nums1: {nums1}')
    print(f'nums2: {nums2}')
    print(f'Median is: {solu.findMedianSortedArrays(nums1, nums2)}')
 
    # test1 of mySqrt:
    x = 8
    print('test1 of mySqrt:')
    print('Input: x={x}')
    print(f'int_sqrt({x}) is: {solu.mySqrt(x)}')
    
    # test2 of mySqrt:
    x = 16
    print('test2 of mySqrt:')
    print('Input: x={x}')
    print(f'int_sqrt({x}) is: {solu.mySqrt(x)}')

