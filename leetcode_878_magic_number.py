# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 10:37:46 2019
leetcode_878_magic_number.py
参考如下blog:
https://blog.csdn.net/XX_123_1_RJ/article/details/81476088    
"""

"""
leetcode_878:
A positive integer is magical if it is divisible by either A or B.
Return the N-th magical number.  Since the answer may be very large, return it modulo 10^9 + 7.
Example 1:
Input: N = 1, A = 2, B = 3
Output: 2
Example 2:
Input: N = 4, A = 2, B = 3
Output: 6
Example 3:
Input: N = 5, A = 2, B = 4
Output: 10
Example 4:
Input: N = 3, A = 6, B = 4
Output: 8
 
Note:
1 <= N <= 10^9
2 <= A <= 40000
2 <= B <= 40000
"""

class Solution:
    # 二分法思路
    def nthMagicalNumber(self, N, A, B):

        def gcd(x, y):  # 求两个数的最大公约数
            return x if not y else gcd(y, x % y)  # 辗转取余法

        mod = 10**9 + 7
        L = A * B // gcd(A, B)  # 两个数的最小公倍数

        def magic_below_n(n):  # 返回能整数n包含 能被A或B整除的数个数
            # 容斥原理：A并B个数 = A个数 + B个数 - A交B个数
            return n // A + n // B - n // L  # n//A表示能被A整除的数，n//B表示能被B整除的数，n//L表示能同时被A或 B整除的数个数(注意这里用的是最小公倍数)

        low, high = 0, 10**15  # 初始化上下界
        while low < high:  # 进行二分查找
            mid = (low + high) // 2
            if magic_below_n(mid) < N:  # low和high的处理公式使得最终找到的是取N值的第一个n
                low = mid + 1
            else:
                high = mid

        return low % mod  # 取模

     # 朴素的循环思路
    def nthMagicalNumber_low_efficient(self, N, A, B):

        mod = 10**9 + 7
        magic_num = []
        n = 1
        while len(magic_num) < N:
            if (n%A == 0) or (n%B == 0):
                magic_num.append(n)
            n += 1

        return (n-1) % mod  # 取模

if __name__ == '__main__':
    
    import time
    
    print('test1 of solution')
    N, A, B = 8, 12, 15
    solu = Solution()
    starttime = time.time()
    print(f'Inputs: A={A}, B={B}, N={N}')
    print(f'Output: magic number is {solu.nthMagicalNumber(N, A, B)}')
    print(f'Running Time:{time.time() - starttime}')
    
    print('test2 of solution')
    N, A, B = 1279467, 128, 180
    solu = Solution()
    starttime = time.time()
    print(f'Inputs: A={A}, B={B}, N={N}')
    print(f'Output: magic number is {solu.nthMagicalNumber(N, A, B)}')
    print(f'Running Time:{time.time() - starttime}')
    
    print('test1 of low_efficient solution')
    N, A, B = 3, 6, 4
    solu = Solution()
    starttime = time.time()
    print(f'Inputs: A={A}, B={B}, N={N}')
    print(f'Output: magic number is {solu.nthMagicalNumber_low_efficient(N, A, B)}')
    print(f'Running Time:{time.time() - starttime}')
    
    print('test2 of low_efficient solution')
    N, A, B = 1279467, 128, 180
    solu = Solution()
    starttime = time.time()
    print(f'Inputs: A={A}, B={B}, N={N}')
    print(f'Output: magic number is {solu.nthMagicalNumber_low_efficient(N, A, B)}')
    print(f'Running Time:{time.time() - starttime}')
