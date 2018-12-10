#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二叉树模型计算期权价格：
参考《金融数学》第2、3章内容
(version1.0)
(18/12/09)

@author: yuxun
"""

import numpy as np
import scipy.stats as st

def option_value_bin(n, T, s_0, X, sigma, r, option_info):
    '''
    Input:
        n:二叉树参数，分成多少期二叉树
        T:期权的存续时间(天数)
        s_0:当前价格
        X:执行价格
        sigma:年化波动率
        r:年化无风险收益率
        option_info:一个存储了option信息的字典
                    option_direction:期权方向，1为看涨，-1为看跌
                    option_type:期权类型，European/American/Barrier（参考《金融数学》定义）
                    barrier_dire/barrier_price：当期权类型为Barrier时，还需要定义改值确定障碍
    Output:
        v_0:期权价格
    '''
    # (0) 加载option基础信息
    # (0.1) option类型信息
    option_direction = option_info['option_direction']  # 期权方向
    option_type = option_info['option_type']  # 期权类型
    if option_type == 'Barrier':
        # 联合起来构成障碍的涵义：
        # barrier_dire=1 & barrier_price=100，价格超过100时敲出
        # barrier_dire=-1 & barrier_price=95，价格低于95时敲出
        barrier_dire = option_info['barrier_dire']  # 敲出方向
        barrier_price = option_info['barrier_price']  # 敲出价格
    # (0.2) 单期二叉树的折现因子
    dt = T/ (n*360)  # 每一个二叉树经历的时间（单位为年）
    B = (np.e)**(-r * dt)  # 折现因子B(每个二叉树的前后期折现率)
    # (0.3) 单期二叉树的u/d
    u = (np.e)**(sigma * np.sqrt(dt))
    d = (np.e)**(-sigma * np.sqrt(dt))
    
    # (1) 计算标的资产的多期二叉树
    # (1.1) 初始化
    tree_len = n + 1
    s_tree = np.zeros((tree_len, tree_len))
    s_tree[0, 0] = s_0  # 第一个价格为当期价格
    # (1.2) 递推方法获得整个二叉树
    for i in range(1, tree_len):
        for j in range(0, tree_len):
            if j == 0:
                s_tree[i, j] = s_tree[i-1, 0] * d
            else:
                s_tree[i, j] = s_tree[i-1, j-1] * u
                  
    # (2) 计算衍生资产的多期二叉树
    # (2.1) 初始化            
    v_tree = np.zeros((tree_len, tree_len))
    q = (1/B - d) / (u - d)  # 风险上涨中性概率
    ### 最后一期的初始化
    ### 最后一期的衍生品价格，都有确定性的公式
    if option_type in ('European', 'American'):
        v_tree[n, ] = np.array([ max(option_direction*(i-X), 0) for i in list(s_tree[n, ]) ])
    elif option_type == 'Barrier':
        v_tree[n, ] = np.array([ 0 if barrier_dire*(i - barrier_price) > 0 else max(option_direction*(i-X), 0) for i in list(s_tree[n, ]) ]) 
        
    # (2.2) 递推方法获得整个二叉树：不同类型期权，有不同的递推公式
    for i in range(n-1, -1, -1):
        for j in range(0, i+1):
            p1 = B * ((1-q)*v_tree[i+1, j] + q*v_tree[i+1, j+1])  # 欧式期权的价格
            # 根据不同的类型，计算相应的衍生品二叉树节点价格
            if option_type == 'European':        
                v_tree[i, j] = p1
            elif option_type == 'American':
                p2 = max(option_direction*(s_tree[i, j]-X), 0)
                v_tree[i, j] = max(p1, p2)
            elif option_type == 'Barrier':
                if barrier_dire*(s_tree[i, j] - barrier_price) > 0:
                    v_tree[i, j] = 0
                else:
                    v_tree[i, j] = p1
                
    # (2.3) 获得衍生品价格        
    v_0 = v_tree[0, 0]
    
    return v_0



# 欧式看涨期权
option_info = {'option_direction': 1,
               'option_type': 'European'
               }
v_0 = option_value_bin(20, 60, 100, 105, 0.2, 0.05, option_info)
print(v_0)
# 美式看涨期权
option_info = {'option_direction': 1,
               'option_type': 'American'
               }
v_0 = option_value_bin(20, 60, 100, 105, 0.2, 0.05, option_info)
print(v_0)
# 欧式看跌期权
option_info = {'option_direction': -1,
               'option_type': 'European'
               }
v_0 = option_value_bin(20, 60, 95, 105, 0.2, 0.05, option_info)
print(v_0)
# 美式看跌期权
option_info = {'option_direction': -1,
               'option_type': 'American'
               }
v_0 = option_value_bin(20, 60, 95, 105, 0.2, 0.05, option_info)
print(v_0)
print(f'max(X-S, 0):{100-90}; max(X*e^(-rt)-S, 0):{100*(np.e)**(-0.05*1/12)-90}')
# 欧式看跌期权
option_info = {'option_direction': -1,
               'option_type': 'European'
               }
v_0 = option_value_bin(20, 3600, 95, 105, 0.2, 0.05, option_info)
print(v_0)
# 障碍期权
option_info = {'option_direction': 1,
               'option_type': 'Barrier',
               'barrier_dire': 1,
               'barrier_price': 110
               }
v_0 = option_value_bin(20, 60, 100, 105, 0.2, 0.05, option_info)
print(v_0)



def option_value_BS(T, s_0, X, sigma, r):
    '''
    作为参考的Black-Scholes公式的结果
    '''
    # 利用输入，计算相关参数
    tau = T/360
    d1 = ( np.log(s_0/X) + (r+sigma**2/2)*tau ) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    # 代入公式
    C = s_0*st.norm.cdf(d1) - X*(np.e)**(-r*tau)*st.norm.cdf(d2)
    P = X*(np.e)**(-r*tau)*st.norm.cdf(-d2) - s_0*st.norm.cdf(-d1)
    return C, P

# 欧式看涨期权
C, P = option_value_BS(60, 100, 105, 0.2, 0.05)
print(f'BM result: call price is {C}')
# 欧式看跌期权
C, P = option_value_BS(60, 95, 105, 0.2, 0.05)
print(f'BM result: put price is {P}')