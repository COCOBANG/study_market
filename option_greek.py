#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Black-Scholes期权公式求出来的greek
(version1.0)
(18/12/09)

@author: yuxun
"""

import numpy as np
import scipy.stats as st

def option_greek(T, s_0, X, sigma, r):
    '''
    Input:
        T:期权的存续时间(天数)
        s_0:当前价格
        X:执行价格
        sigma:年化波动率
        r:年化无风险收益率
    Output:
        greek_C:看涨期权的greek字典
        greek_P:看跌期权的greek字典
    '''
    # 利用输入，计算相关参数
    tau = T/360
    d1 = ( np.log(s_0/X) + (r+sigma**2/2)*tau ) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    # 代入看涨期权greek公式
    delta_C = st.norm.cdf(d1)
    gamma_C = (np.e)**(-d1**2/2) / ( s_0*sigma*np.sqrt(2*np.pi*tau) )
    vega_C = s_0 * (np.e)**(-d1**2/2) * np.sqrt(tau/(2*np.pi))
    theta_C = r*X*(np.e)**(-r*tau)*st.norm.cdf(d2) + ( s_0*sigma*(np.e)**(-d1**2/2) ) / ( 2*np.sqrt(2*np.pi*tau) )
    rho_C = tau*X*(np.e)**(-r*tau)*st.norm.cdf(d2)
    greek_C = {'delta': delta_C,
               'gamma': gamma_C,
               'vega': vega_C,
               'theta': theta_C,
               'rho': rho_C}
    # 代入看跌期权greek公式
    delta_P = delta_C - 1
    gamma_P = gamma_C
    vega_P = vega_C
    theta_P = theta_C - r*X*(np.e)**(-r*tau)
    rho_P = rho_C - tau*X*(np.e)**(-r*tau)
    greek_P = {'delta': delta_P,
               'gamma': gamma_P,
               'vega': vega_P,
               'theta': theta_P,
               'rho': rho_P}
    # 返回值
    return greek_C, greek_P

# 产生一个gamma和vega不一致的组合
# 一个快要到期的ATM期权1
greek_C1, greek_P1 = option_greek(5, 10, 10, 0.2, 0.05)
# 一个到期时间长一点的期权2
greek_C2, greek_P2 = option_greek(60, 10, 10, 0.2, 0.05)
# 做多1个期权1，做空1个期权2
gamma_C1, gamma_C2 = greek_C1['gamma'], greek_C2['gamma']
gamma_port = gamma_C1 - gamma_C2
vega_C1, vega_C2 = greek_C1['vega'], greek_C2['vega']
vega_port = vega_C1 - vega_C2
print(f'option1: gamma({gamma_C1}); vega({vega_C1})')
print(f'option2: gamma({gamma_C2}); vega({vega_C2})')
print(f'portfolio(long option1 short option2): gamma({gamma_port}); vega({vega_port})')
# 画一下组合的gamma和vega随着股价变动的图像
from matplotlib import pyplot as plt
price = np.arange(8, 12, 0.1)
gamma = [option_greek(5, s, 10, 0.2, 0.05)[0]['gamma'] - option_greek(60, s, 10, 0.2, 0.05)[0]['gamma'] for s in price]
vega = [option_greek(5, s, 10, 0.2, 0.05)[0]['vega'] - option_greek(60, s, 10, 0.2, 0.05)[0]['vega'] for s in price]
# 画图
plt.figure(figsize=(6, 4))
plt.grid()
plt.xlabel('stock price')
plt.ylabel("option portfolio' greek")
plt.plot(price, gamma, '.g-', label="portfolio gamma")
plt.plot(price, vega, '.r--', label="portfolio vega")
plt.legend()