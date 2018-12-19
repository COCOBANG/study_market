#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 10:03:51 2018

@author: yuxun
"""

import pandas as pd
import numpy as np
import ccxt
import time
import matplotlib.pyplot as plt
from sklearn import linear_model


# ---------------------- (〇) 全部有用的函数 ----------------------


def Strategy_Stat(x, Y, freq):
    """
    计算回测统计指标的函数：
    Stat, total = Strategy_Stat(x, Y, freq)
    
    Input:
        x:一个DataFrame，每一个column是一个signal，index是时间点
        Y:一个DataFrame，index是时间点，只有一个column，是一个命名为'frr'的值，表示未来一段的收益
        freq:Y的时段的频率，'nd'表示n天，'nh'表明n小时，'nm'表明n分钟
    Output:
        Stat:一个统计值的DataFrame，index是输入x的signal名称，每一个column表示一种统计值
        total:收益和回撤的全样本数据
    """
    # --合并x、Y，并计算一些常数--
    total = pd.concat([Y, x], axis=1, join='inner', sort=False)
    columns = list(x.columns)  # x的字段名
    columns_DD = [i + '_DrawDown' for i in columns]  # x+'drawdown'的字段名
    num = int(freq[:-1])  # freq的参数
    date = [x[0:10] for x in list(total.index)]  # 将日期设置为index
    total.index = date
    # --计算SR和Calmar比率--
    # 先求日内级别的日收益率
    if ('h' in freq) or ('m' in freq):
        for xName in columns:
            total[f'{xName}'] = 1 * (total[f'{xName}'] > 0) - 1 * (total[f'{xName}'] < 0)  # 首先将信号转为-1,0,1
            total[f'{xName}'] = total['frr'] * total[f'{xName}']
        total = total[columns].groupby(total.index).sum()
    # 再求相关的策略统计量
    length = len(total)  # 数据长度
    date = total.index  # total的index
    for xName in columns:
        # 先求一下日级别的日收益率(前面先求了日内级别的日收益率)
        if 'd' in freq:  # 日级别的，综合signal可以认为是过去num天的均值
            total[f'{xName}'] = 1 * (total[f'{xName}'] > 0) - 1 * (total[f'{xName}'] < 0)  # 首先将信号转为-1,0,1
            total[f'pos_{xName}'] = total[f'{xName}'].rolling(num, min_periods=1).sum() / num  # 过去num天的均值计算综合signal
            total[f'{xName}'] = total['frr'] * total[f'pos_{xName}']
        # 策略的累计收益和回撤
        total[f'{xName}'] = total[f'{xName}'].fillna(0)  # 用0填充nan
        Cum = np.zeros(length)
        MaxToDate = np.zeros(length)
        DrawDown = np.zeros(length)
        for i in range(length):
            if i == 0:
                Cum[i] = total[f'{xName}'][0]
                MaxToDate[i] = max(0, Cum[i])
            else:
                Cum[i] = Cum[i-1] + total[f'{xName}'][i]
                MaxToDate[i] = max(MaxToDate[i-1], Cum[i])
            DrawDown[i] = MaxToDate[i] - Cum[i]
        total = pd.merge(total, pd.DataFrame(DrawDown, columns=[f'{xName}_DrawDown'], index=date), left_index=True, right_index=True) 
    # 策略收益均值、标准差和最大回撤
    ret = np.array(total[columns].mean()) * 1e4  # 日均收益(单位:bps)
    ret_AP = np.array(total[columns].mean()) * 365  # 年化收益
    std_AP = np.array(total[columns].std()) * np.sqrt(365)  # 年化标准差
    maxDD = np.array(total[columns_DD].max())  # 最大回撤
    # 策略的统计数据
    SR = ret_AP / std_AP  # SR
    Calmar = ret_AP / maxDD  # Calmar
    Stat = np.hstack((ret.reshape(len(ret), 1), ret_AP.reshape(len(ret_AP), 1), maxDD.reshape(len(maxDD), 1), SR.reshape(len(SR), 1), Calmar.reshape(len(Calmar), 1)))
    Stat = pd.DataFrame(Stat, index=columns, columns=['ret', 'ret_AP', 'maxDD', 'SR', 'Calmar'])
    
    return Stat, total


def plot(total, signalName):
    """
    基于total的画图函数：
    plot(total, signalName)
    
    Input:
        total:Strategy_Stat函数得到的total结果DataFrame
        signalName:查看哪个具体的signal的名称
    """
    plot = total[[f'{signalName}']].cumsum()
    # 画图的设置
    fig, ax = plt.subplots()
    xticks = range(0, len(plot), 50)
    xlabels = [plot.index[i] for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90)
    plt.title(f'signal:{signalName}')
    # 画图
    plt.plot(plot[f'{signalName}'])
    plt.grid()
    plt.show()    


# ---------------------- (一) 需要预测的Y ----------------------
# 从ccxt抓取数据
proxies = {'https': 'http://127.0.0.1:1087'}  # 代理的IP
binance=ccxt.binance()
binance.proxies = proxies

# 需要预测的Y
BE = binance.fetch_ohlcv('ETH/BTC', timeframe="1d", limit=1000)
timestamp = [i[0] for i in BE]
BEdata = [i[1:] for i in BE]
BEdf = pd.DataFrame(BEdata, columns=["Open", "High", "Low", "Close", "Volume"], index=timestamp)
BEdf.index = [time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(i/1000)) for i in BEdf.index]

BEdf_Y = BEdf[['Close']]
# BEdf_Y.plot()
BEdf_Y['Close_lead_1'] = BEdf_Y[['Close']].shift(-1)
BEdf_Y['Close_lead_5'] = BEdf_Y[['Close']].shift(-5)
BEdf_Y['frr'] = np.log(BEdf_Y['Close_lead_1'] / BEdf_Y['Close'])
BEdf_Y['frr5'] = np.log(BEdf_Y['Close_lead_5'] / BEdf_Y['Close'])
BEdf_Y = BEdf_Y[['frr', 'frr5']]


# ---------------------- (二) 各种x的遍历 ----------------------
# 第五个x:过去n根k线收益的偏度
signal5 = BEdf[['Close']].copy()
signal5['rr'] = np.log(signal5.pct_change() + 1)
signal5 = signal5[['rr']]
for i in range(5, 61, 5):
    # 首先计算过去i天的收益偏度
    x = signal5[['rr']].rolling(window = i).skew()
    x[f'rr_skew_{i}'] = x['rr']
    x = x[[f'rr_skew_{i}']]
    for j in range(5, 121, 5):
        # 然后看它和过去j天的均值的关系
        y = x.rolling(window = j).mean().rename(columns={f'rr_skew_{i}': f'rr_skew_{i}_mean{j}'})
        y = pd.concat([x, y], axis=1).dropna()
        signal5[f'rr_skew_{i}_{j}'] = 1 * (y[f'rr_skew_{i}'] >= y[f'rr_skew_{i}_mean{j}']) - 1 * (y[f'rr_skew_{i}'] < y[f'rr_skew_{i}_mean{j}'])

columns5 = list(filter(lambda x: x[0:7] == 'rr_skew', list(signal5.columns)))
signal5 = -signal5[columns5]

Stat5, total5 = Strategy_Stat(signal5, BEdf_Y, '5d')
Stat5 = Stat5.sort_values('SR')

plot(total5, 'rr_skew_55_5')