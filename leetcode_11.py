# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:44:41 2019

@author: Administrator
"""

class Solution():
    def maxArea(self, height):
        lft = 0
        rgt = len(height) - 1
        contain = 0
        while lft < rgt:
            contain = max(contain, min(height[lft], height[rgt])*(rgt - lft))
            if height[lft] < height[rgt]:
                lft += 1
            else:
                rgt -= 1
        return contain
    
    def twoSum(self, numbers, target):
        if len(numbers) <= 1:
            return [None, None]
        else:
            lft = 0
            rgt = len(numbers) - 1
            while lft < rgt:
                sumofnum = numbers[lft] + numbers[rgt]
                if sumofnum < target:
                    lft += 1
                elif sumofnum > target:
                    rgt -= 1
                else:
                    return [lft+1, rgt+1]
            return [None, None]
        
    def reverseVowels(self, s):
        vowels = 'aeiouAEIOU'
        cha = [i for i in s]
        if len(s) <= 1:
            return s
        else:
            lft = 0
            rgt = len(s) - 1
            while lft < rgt:
                if cha[lft] not in vowels:
                    lft += 1
                if cha[rgt] not in vowels:
                    rgt -= 1
                if cha[lft] in vowels and cha[rgt] in vowels:
                    t = cha[lft]
                    cha[lft] = cha[rgt]
                    cha[rgt] = t
                    lft, rgt = lft+1, rgt-1
            return "".join(cha)
                


if __name__ == '__main__':
    solu = Solution()
    
    height = [1,8,6,2,5,4,8,3,7]
    print('test of maxArea:')
    print(f'Input: {height}')
    print(f'The Most Contain is : {solu.maxArea(height)}')
    
    numbers, target = [1, 2, 7, 11, 15], 13
    print('test of twoSum:')
    print('Inputs:')
    print(f'numbers: {numbers}')
    print(f'target: {target}')
    print(f'Index: {solu.twoSum(numbers, target)}')
    
    s = 'leetcode'
    solu.reverseVowels(s)