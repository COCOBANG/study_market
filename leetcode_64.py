# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:45:35 2019

@author: Administrator
"""

import numpy as np

class Solution:
    def minPathSum(self, grid):
        inf = float('inf')
        grid1 = np.array(grid)
        m, n = grid1.shape
        cost = np.ones(grid1.shape) * inf
        cost[0][0] = grid[0][0]
        cost_ravel = cost.ravel()
        
        processed = []
        while len(processed) < grid1.size:
            low_cost = min(cost_ravel[[i for i in range(m*n) if i not in processed]])
            i_gp, j_gp = np.where(cost == low_cost)
            for x in range(len(i_gp)):
                i = i_gp[x]
                j = j_gp[x]
                # 邻居点的更新
                if i < (m-1) or j < (n-1):  # [m,n]没有邻居，不用更新邻居点
                    if i == (m-1):
                        cost[i, j+1] = min(cost[i, j+1], low_cost + grid1[i, j+1])
                    elif j == (n-1):
                        cost[i+1, j] = min(cost[i+1, j], low_cost + grid1[i+1, j])
                    else:
                        cost[i, j+1] = min(cost[i, j+1], low_cost + grid1[i, j+1])
                        cost[i+1, j] = min(cost[i+1, j], low_cost + grid1[i+1, j])
                cost_ravel = cost.ravel()
                processed.append(i*n+j)
        return int(cost_ravel[-1])

    def minPathSum_nonp(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        grid = grid.copy()
        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                if i != 0 and j == 0:
                    grid[i][j] += grid[i - 1][j]
                if i == 0 and j != 0:
                    grid[i][j] += grid[i][j - 1]
                if i != 0 and j != 0:
                    grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        return grid[m - 1][n - 1]               
                


solu = Solution()                
grid = [
  [1,3,1,2],
  [1,5,1,5],
  [4,2,1,3]
]
print(solu.minPathSum(grid))
print(solu.minPathSum_nonp(grid))