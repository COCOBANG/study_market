# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 12:25:00 2019

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

n = 50
N = 1000
x = np.linspace(-3, 3, n)
X = np.linspace(-9, 9, N)
pix = np.pi * x
y = np.sin(pix) / pix + 0.1*x + 0.05*np.random.randn(n)

# 随机梯度下降算法
hh = 2*0.3**2
theta0 = np.random.randn(n)
e = 0.1
for o in range(n*N):
    i = np.random.randint(0, n-1)
    ki = np.exp(-(x - x[i])**2 / hh)
    theta =  theta0 - e*ki*(np.dot(ki.T, theta0) - y[i])
    if sum((theta-theta0)*(theta-theta0)) < 1e-6:
        break
    theta0 = theta

# 拟合结果
M1 = np.stack([X**2 for i in range(n)], axis=1)
M2 = np.stack([x**2 for i in range(N)], axis=0)
M3 = 2*np.dot(X.reshape((N,1)), x.reshape(1,n))
K = np.exp(-(M1+M2-M3) / hh)
F = np.dot(K, theta)

plt.plot(x, y, 'bx')
plt.plot(X, F, '-r')



n = 200
a = np.linspace(0, 4*np.pi, n//2).reshape((-1, 1))
u = np.row_stack((a*np.cos(a), (a+np.pi)*np.cos(a))) + np.random.randn(n).reshape((-1, 1))
v = np.row_stack((a*np.sin(a), (a+np.pi)*np.sin(a))) + np.random.randn(n).reshape((-1, 1))
x = np.column_stack((u, v))
y = np.column_stack((np.ones((n//2, 1)), -np.ones((n//2, 1))))

y1 = [a*np.cos(a)]