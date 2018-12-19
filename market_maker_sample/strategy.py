#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market_Maker_Sample_Code

@author: yuxun
"""
import traceback
import time
import datetime
import logging
import pandas as pd
import numpy as np
from pandas.tseries.offsets import *
import async_timeout
import asyncio
from functools import partial
from sqlalchemy import create_engine
import aiopg
### 这一部分需要quant_process库
from quant_process import dex
from quant_process import Cache
from quant_process import AccountManager, Account
from quant_process import Subscribe, CurrentData
from quant_process.log import hf_rev_logger as logger

# --输入参数--
ACCOUNTS = []

# 是否用ETH为benchmark
IfUsingETHasBM = None
# 是否用自身历史数据获得信息
IfNotUsingHisData = None
# 做市的pair----需要注意，后面的变量名要避开symbol
symbol = None
# 做市的交易所
MMmarket = None
# benchmark交易所
benchmark = None
# benchmark备用交易所
benchmark2 = None
# benchmark交易所读数timeout的最大数(超过则转向备用交易所)
benmarkDataCountMax = None
# benchmark的pair
symbol_BM = None
# 历史数据获取的时间窗口
TimeWindow = None
# 阈值判断标准差的系数(越小阈值越低，更新越频繁)
StdCo = None
# spread和benchmark历史spread的比例
spread2BM = None
# 最优买卖价位上的挂单量，平均的挂单金额(单位:ETH)
BestPriceOrderNum = None
# 非最优买卖价位上的挂单量，最大的挂单金额(单位:ETH)
nonBestPriceOrderNumMax = None
# 库存调整系数(当库存差额累计的绝对值为该值时，中间价调整1%)
InventoryAdjustCo = None
# 合宜的库存量(单位:个/多少个山寨币)
InventoryWant2be = None
# 一个常数预测量，直接加在Fairvalue的计算式中
AlphaAdjustConstant = None
# 自成交频率(每多少分钟自成交一次)
SelfTradeFreq = None
# 自成交总额的上限(下限是0.11)
SelfTradeTotalMax = None
# 自成交时slave账户山寨币总库存和现金总库存的最低值，master账户山寨币可用和现金可用的最低值(都是以ETH为计价单位)
Slave_Inventory_Total_MIN = None
Slave_Cash_Total_MIN = None
Master_Inventory_free_MIN = None
Master_Cash_free_MIN = None
# 手续费参数
TakerOrderCost = None
MakerOrderCost = None

# --使用ETH为benchmark时的附加参数--
# 山寨币波动和ETH/USDT的相关系数
b = None
# 做市初始的ETH价格
ETH0 = None
# 做市初始的山寨币价格(请自行设定)
Price0 = None

# --不使用自身历史数据获得的信息时附加参数--
# 价差比例，设定的价差是中间价的多少比例
spreadRatio0 = None
# 初始价差，一开始设定的价差是多少(注意：如果选择量使用ETH作为benchmark，因为在那里设定量初始价格，因此PDiff0一定是0)
PDiff0 = None
# 历史数据用的symbol，可以随便用一个有完整历史数据的
symbol_HisData = None

# --全局变量初始值--
# 5分钟采样缓存
cache = None
# orderbook更新次数计数
orderbookUpdateCount = 0
# Master最近一次挂出来的ask1和bid1的价格，一个缓存数据
BestPrice_Master = Cache([[1000, 1e-8]], columns=['ask1', 'bid1'])
# orderbook更新是否正常(正常为0，不正常为1)，不正常时，下一轮循环要继续更新orderbook
ISOrderbookUpdateNotOK = 0
# spread放宽系数，不存在放宽情况时为1(spread不放宽)，出现放宽情况时大于1(spread*spreadWidenCoef)
spreadWidenCoef = 1
# 相关的网络地址
db_url = ""
redis_url = ""


# ----(0)读取历史数据----
# 从数据库读数的函数

def db_sql():
    logger.info('"msg": "db init!"')
    # ----从数据库读取数据----
    # --sql读数--
    dt_begin = datetime.datetime.now() - (TimeWindow + 8) * Hour()  # datetime是0时区，now()是东8区
    dt_begin_str = dt_begin.strftime("%Y-%m-%d %H:%M:%S")
    # 数据库连接
    engine = create_engine(db_url)
    if IfNotUsingHisData == 0:
        symbol_sql = symbol
    else:
        symbol_sql = symbol_HisData
    sql = f"""
      SELECT *
      FROM
      (
      SELECT trade_market,symbol,datetime,date,CAST(asks->0->>0 as NUMERIC) as ask1,CAST(bids->0->>0 as NUMERIC) as bid1
      FROM orderbook
      WHERE ((symbol='{symbol_sql}' and trade_market='{MMmarket}') or (symbol='{symbol_BM}' and trade_market='{benchmark}'))
      and datetime>='{dt_begin_str}'
      ) as a
      WHERE mod(extract(minute from datetime)::int,5)=0
      """
    df = pd.read_sql(sql, engine)
    logger.info('"msg": "db init get data!"')
    # --预处理:时间对齐--
    # 首先对值进行lead、lag处理
    df1 = df.drop(['symbol', 'date'], axis=1).sort_values('datetime')
    df1 = df1.dropna(axis=0, how='any')  # 去掉包含了NAN的行
    df1_lag1, df1_lead1 = df1.shift(1), df1.shift(-1)
    df1_lag1, df1_lead1 = df1_lag1.rename(columns=lambda x: x + '_lag1'), df1_lead1.rename(columns=lambda x: x + '_lead1')
    df1 = pd.concat([df1, df1_lag1, df1_lead1], axis=1).dropna(axis=0, how='any')
    # 每一行判断lead、lag哪个更近，用更近的ask1和bid1作为benchmark(尽可能对齐)
    def neardt(arrLike):
        trade_market, datetime, datetime_lag1, datetime_lead1 = arrLike[['trade_market', 'datetime', 'datetime_lag1', 'datetime_lead1']]
        ask1_lag1, bid1_lag1, ask1_lead1, bid1_lead1 = arrLike[['ask1_lag1', 'bid1_lag1', 'ask1_lead1', 'bid1_lead1']]
        trade_market_lag1, trade_market_lead1 = arrLike[['trade_market_lag1', 'trade_market_lead1']]
        if trade_market == MMmarket:
            if trade_market_lag1 == benchmark and trade_market_lead1 == benchmark:  # 前后都是benchmark，比较时间远近
                if abs(datetime - datetime_lag1) <= abs(datetime - datetime_lead1):
                    ask1, bid1 = ask1_lag1, bid1_lag1
                else:
                    ask1, bid1 = ask1_lead1, bid1_lead1
            elif trade_market_lag1 != benchmark and trade_market_lead1 == benchmark:  # lag1的不是benchmark，则直接用lead1赋值
                ask1, bid1 = ask1_lead1, bid1_lead1
            elif trade_market_lag1 == benchmark and trade_market_lead1 != benchmark:  # lead1的不是benchmark，则直接用lag1赋值
                ask1, bid1 = ask1_lag1, bid1_lag1
            else:  # lag1和lead1都不是benchmark，则不赋值
                ask1, bid1 = np.NaN, np.NaN
        else:
            ask1, bid1 = np.NaN, np.NaN
        return ask1, bid1
    df1['BM'] = df1.apply(neardt, axis=1)
    df1['ask1_BM'] = df1['BM'].apply(lambda x: x[0])
    df1['bid1_BM'] = df1['BM'].apply(lambda x: x[1])
    # 只看dex且前后至少有一个benchmark的观测，保留下需要的数据
    OBdf = df1[(df1['trade_market'] == MMmarket) & ((df1['trade_market_lag1'] != MMmarket) | (df1['trade_market_lead1'] != MMmarket))]
    OBdf = OBdf[['datetime', 'ask1', 'bid1', 'ask1_BM', 'bid1_BM']]
    OBdf = OBdf.reset_index().drop(columns='index')
    OBdf = OBdf.set_index('datetime')
    global cache
    cache = Cache(OBdf.values.tolist(), columns=['ask1', 'bid1', 'ask1_BM', 'bid1_BM'])
    logger.info('"msg": "db init cache data!"')
    # 获得一个全局Cache类，记录了dex和benchmark的的历史ask1、bid1信息


# db_sql()
    

# ----(1)main函数中用到的各个函数的定义----
"""
两个数据转换函数：
(1)如果IfUsingETHasBM=1，即用ETH作为benchmark，则需要做出一个benchmark的模拟价格(历史值和当前值都要转换)
(2)如果IfNotUsingHisData=1，即不在历史数据中学习价差规律，则需要做出一个dex的模拟价格(只需要转换历史值)
(3)如果IfUsingETHasBM=1且IfNotUsingHisData=1，需要先做benchmark模拟，再做dex模拟
"""

def data_transform_RealTime(msg, msg_BM):
    """
    处理当前实时数据，只有benchmark的值可能转换
    
    OB_RealTime = data_transform_RealTime(msg,msg_BM)
    Input:
    msg:dex的实时数据
    msg_BM:bemchmark交易所的实时数据
    Ouput:
    OB_RealTime:一个dataframe，根据条件对benchmark的当前价格转换，然后获得的一个ask1、bid1、ask1_BM、bid1_BM四列数的表
    """
    # --benchmark数据--
    ordertime_BM = msg_BM.get("datetime")
    symbol_BM = msg_BM.get("symbol")
    ask1_BM = msg_BM.get("asks")[0][0]
    bid1_BM = msg_BM.get("bids")[0][0]
    # 以ETH/USDT作为benchmark，需要将ETH/USDT转换为一个模拟价格：(1-b)*山寨币初始价格+(山寨币初始价格*b/ETH初始价格)*ETH价格
    ETH2BMPrice = b * Price0 / ETH0  # 转换系数
    if IfUsingETHasBM == 1:
        ask1_BM = (1 - b) * Price0 + ETH2BMPrice * ask1_BM
        bid1_BM = (1 - b) * Price0 + ETH2BMPrice * bid1_BM
    midP_BM = (ask1_BM + bid1_BM) / 2
    # --dex数据--
    ordertime = msg.get("datetime")
    symbol = msg.get("symbol")
    ask1 = msg.get("asks")[0][0]
    bid1 = msg.get("bids")[0][0]
    midP = (ask1 + bid1) / 2
    logger.info(f'"data": "dex:[{ordertime}][{symbol}]-ask1:{round(ask1,8)}-bid1:{round(bid1,8)}-midP:{round(midP,8)}"')
    logger.info(f'"data": "benchmark:[{ordertime_BM}][{symbol_BM}]-ask1:{round(ask1_BM,8)}-bid1:{round(bid1_BM,8)}-midP:{round(midP_BM,8)}"')
    # --合并生成一个dataframe--
    x = datetime.datetime.utcfromtimestamp(msg.get("nonce"))
    y = x.strftime("%Y-%m-%d %H:%M:%S")
    UTCordertime = datetime.datetime.strptime(y, "%Y-%m-%d %H:%M:%S")
    OB_RealTime = pd.DataFrame({'ask1': ask1, 'bid1': bid1, 'ask1_BM': ask1_BM, 'bid1_BM': bid1_BM}, index=[UTCordertime])
    return OB_RealTime


def data_transform_His(cache):
    """
    处理历史数据，benchmark和dex的值都可能要转换
    
    OBdf = data_transform_His(cache)
    Input:
    cache:缓存的过去一段时间的历史数据
    Ouput:
    OBdf:一个dataframe，根据条件对benchmark和dex的历史价格转换，然后获得的一个ask1、bid1、ask1_BM、bid1_BM四列数的表
    """
    # --取数据--
    OBdf = cache.to_dataframe()
    # --历史orderbook数据不够的报警(可能是数据库的bug)--
    if len(OBdf) < ((TimeWindow * 12) * 0.5):
        logger.error(f'"error": "Hisdata OBdf has {len(OBdf)} rows, TimeWindow should have {TimeWindow*12} rows, Please Check DataBase!"')
    # --benchmark数据--
    # 以ETH/USDT作为benchmark，需要将ETH/USDT转换为一个模拟价格：(1-b)*山寨币初始价格+(山寨币初始价格*b/ETH初始价格)*ETH价格
    ETH2BMPrice = b * Price0 / ETH0  # 转换系数
    if IfUsingETHasBM == 1:
        OBdf['ask1_BM'] = (1 - b) * Price0 + ETH2BMPrice * OBdf['ask1_BM']
        OBdf['bid1_BM'] = (1 - b) * Price0 + ETH2BMPrice * OBdf['bid1_BM']
    # --dex数据--
    # 不用历史数据时，需要构造一个虚拟的山寨币历史数据，使得历史数据满足spread0、PDiff0
    if IfNotUsingHisData == 1:
        OBdf['midP0'] = (OBdf['ask1_BM'] + OBdf['bid1_BM']) / 2 + PDiff0  # 模拟的中间价：benchmark的中间价加上给定价差PDiff0
        OBdf['spread0'] = OBdf['midP0'] * spreadRatio0  # 模拟的spread0：中间价乘以给定的价差比例spreadRatio0
        OBdf['ask1_rand'] = np.random.uniform(-1 / 2, 1 / 2, len(OBdf))  # 计算ask1和bid1，加上了一点随机量
        OBdf['bid1_rand'] = np.random.uniform(-1 / 2, 1 / 2, len(OBdf))
        OBdf['ask1'] = OBdf['midP0'] + (1 / 2 + OBdf['ask1_rand']) * OBdf['spread0']
        OBdf['bid1'] = OBdf['midP0'] - (1 / 2 + OBdf['bid1_rand']) * OBdf['spread0']
        OBdf = OBdf[['ask1', 'bid1', 'ask1_BM', 'bid1_BM']]
    # --剔除明显错误数据(3个标准差以外的值删除)--
    OBdf_mean = OBdf.mean()
    OBdf_std = OBdf.std()
    bid1_sup, bid1_inf = OBdf_mean['bid1'] + 3 * OBdf_std['bid1'], OBdf_mean['bid1'] - 3 * OBdf_std['bid1']
    ask1_sup, ask1_inf = OBdf_mean['ask1'] + 3 * OBdf_std['ask1'], OBdf_mean['ask1'] - 3 * OBdf_std['ask1']
    bid1_BM_sup, bid1_BM_inf = OBdf_mean['bid1_BM'] + 3 * OBdf_std['bid1_BM'], OBdf_mean['bid1_BM'] - 3 * OBdf_std['bid1_BM']
    ask1_BM_sup, ask1_BM_inf = OBdf_mean['ask1_BM'] + 3 * OBdf_std['ask1_BM'], OBdf_mean['ask1_BM'] - 3 * OBdf_std['ask1_BM']
    OBdf = OBdf[(bid1_inf < OBdf['bid1']) & (OBdf['bid1'] < bid1_sup) & (ask1_inf < OBdf['ask1']) & (OBdf['ask1'] < ask1_sup)]  # 自己交易所的筛选
    OBdf = OBdf[(bid1_BM_inf < OBdf['bid1_BM']) & (OBdf['bid1_BM'] < bid1_BM_sup) & (ask1_BM_inf < OBdf['ask1_BM']) & (OBdf['ask1_BM'] < ask1_BM_sup)]  # benchmark交易所的筛选

    logger.info(f'"data": "Get Hisdata OBdf, the last row is: {OBdf.iloc[-1]}"')

    return OBdf


async def calcul_balance(account_manager):
    """
    计算做市商账户balance信息的函数：
    生成一个dataframe，包含各个账户的cash(ETH)和山寨币库存情况
    
    balance = await calcul_balance(account_manager)
    Input:
    account_manager:账户管理类
    Output:
    balance:做市商账户组的账户信息dataframe
    """
    # 登录账户并读取数据
    logger.info('"msg": "calcul_balance"')
    try:
        await account_manager.login()
    except:
        logger.error('"error": "Account Login Timeout!"')
        raise
    
    total_balance = await account_manager.total_balance()
    logger.info('"msg": "read balance done"')

    InventorySymbol = symbol.split('/')[0]  # 山寨币的symbol
    
    balance = pd.DataFrame(total_balance).T
    balance['cash_total'] = balance['ETH'].apply(lambda x: float(x['total']))
    balance['cash_free'] = balance['ETH'].apply(lambda x: float(x['free']))
    balance['Inventory_total'] = balance[InventorySymbol].apply(lambda x: float(x['total']))
    balance['Inventory_free'] = balance[InventorySymbol].apply(lambda x: float(x['free']))
    balance = balance[['cash_total', 'cash_free', 'Inventory_total', 'Inventory_free']]
    return balance


def calcul_spread(OBdf, OB_RealTime):
    """
    计算spread的函数：
    (1) 如果不用ETH/USDT作为benchmark(其他交易所有活跃品质)，则用benchmark交易所的spread(max(当前,历史))，乘以一个扩大比例系数;
    (2) 如果用ETH/USDT作为benchmark，则benchmark的spread不能用了(因为ETH/USDT很活跃,spread很小)，则用自己的历史spread
    
    spread = calcul_spread(OBdf)
    Input:
    OBdf:处理后的历史orderbook数据(5分钟采样)
    OB_RealTime:当前实时的ask1/bid1数据
    Output:
    spread:最优买卖的价差
    """
    spread_Mean = (OBdf['ask1'] - OBdf['bid1']).mean()
    spread_BM_RT_Mean = (OB_RealTime['ask1_BM'][0] - OB_RealTime['bid1_BM'][0])
    spread_BM_Mean = (OBdf['ask1_BM'] - OBdf['bid1_BM']).mean()
    # 依据条件判断用benchmark的spread还是自己的历史spread
    if IfUsingETHasBM == 0:
        spread = max(spread_BM_RT_Mean, spread_BM_Mean) * spread2BM
    else:
        spread = spread_Mean

    logger.info(f'"data": "Set Best bid/ask Spread: {spread}"')
    return spread


def calcul_fairvalue(balance, InventoryWant2be, OBdf, OB_RealTime):
    """
    计算fairvalue的函数：
    fairvalue = benchmark中间价 + 价差的移动均值 + 库存调整项Adjust1 + 预测调整项Adjust2
    其中：
    Adjust1 = ((与目标库存的差)*(benchmark中间价)/激进系数)*(中间价的1%)
    
    Fairvalue = calcul_fairvalue(balance,InventoryWant2be,OBdf,OB_RealTime)
    Input:
    balance:做市商账户的持仓情况
    InventoryWant2be:依据自成交情况更新的山寨币合宜库存
    OBdf:处理后的历史orderbook数据(5分钟采样)
    OB_RealTime:当前实时的ask1/bid1数据
    Output:
    Fairvalue:公平的中间价
    """
    # benchmark的中间价
    midP_BM_RT = float((OB_RealTime['ask1_BM'] + OB_RealTime['bid1_BM']) / 2)
    # 价差的移动均值
    OBdf['midPDiff'] = (OBdf['ask1'] + OBdf['bid1'] - OBdf['ask1_BM'] - OBdf['bid1_BM']) / 2
    HisMean = OBdf['midPDiff'].mean()
    # 库存调整项
    InventorySum = balance['Inventory_total'].sum()
    Adjust1 = -((InventorySum - InventoryWant2be) * midP_BM_RT / InventoryAdjustCo) * (1e-2 * midP_BM_RT)
    logger.info(f'"data": "InventoryWant2be:{InventoryWant2be}--InventorySum:{InventorySum}--InventoryDiff(ETH):{(InventorySum-InventoryWant2be)*midP_BM_RT}"')
    # 预测调整项
    Adjust2 = AlphaAdjustConstant
    Fairvalue = midP_BM_RT + HisMean + Adjust1 + Adjust2
    logger.info(f'"data": "Fairvalue({round(Fairvalue,8)}) = BenchmarkPrice({round(midP_BM_RT,8)}) + MA_PDiff({round(1e8*HisMean,2)}*1e-8) + Adjust1({round(Adjust1,8)}) + Adjust2({round(Adjust2,8)})"')
    return Fairvalue


def calcul_result(OBdf, OB_RealTime, BestPrice_Master, Fairvalue):
    """
    计算Result的函数：
    如果当前的中间价低于(Fairvalue-阈值)，则Result=1，表明需要向上重置orderbook
    如果当前的中间价高于(Fairvalue+阈值)，则Result=-1，表明需要向下重置orderbook
    其余情况，Result=0
    其中，中间价的计算需要注意，我们会忽略那些在Master最近一次挂的(bid1*,ask1*)区间内的价格，认为这些挂单不影响Fairvalue的判断；
    也就是说，如果最新的bid1_RT满足bid1*<bid1_RT<ask1*，则强制让bid1=bid1*，其余则是一般情形所认为的bid1=bid1_RT(认为挂在区间内的单子是噪声)
    同样的，如果最新的ask1_RT满足bid1*<ask1_RT<ask1*，则强制让ask1=ask1*，其余则是一般情形所认为的ask1=ask1_RT
    
    Result = calcul_result(OBdf,OB_RealTime,BestPrice_Master,Fairvalue)
    Input:
    OBdf:处理后的历史orderbook数据(5分钟采样)
    OB_RealTime:当前实时的ask1/bid1数据
    BestPrice_Master:最近一次Master挂的ask1/bid1的价格
    Fairvalue:公平的中间价
    Output:当前实时的ask1/bid1数据
    Result:结果变量
    """
    # 计算调整的dex实时中间价midP_RT_Adjust(依据实时ask1/bid1和最近一次Master挂出来的ask1/bid1相对关系来调整)
    ask1_Master = BestPrice_Master.to_dataframe()['ask1'][0]
    bid1_Master = BestPrice_Master.to_dataframe()['bid1'][0]
    ask1_RT = OB_RealTime['ask1'][0]
    bid1_RT = OB_RealTime['bid1'][0]
    if (bid1_Master < bid1_RT and bid1_RT < ask1_Master):
        bid1 = bid1_Master
    else:
        bid1 = bid1_RT
    if (bid1_Master < ask1_RT and ask1_RT < ask1_Master):
        ask1 = ask1_Master
    else:
        ask1 = ask1_RT
    midP_RT_Adjust = round((ask1 + bid1) / 2, 8)
    logger.info(f'"data": "Ajusted dex:[{symbol}]-ask1:{round(ask1,8)}-bid1:{round(bid1,8)}-midP_Adjust:{round(midP_RT_Adjust,8)}"')
    # dex_benchmark价差的历史标准差
    OBdf['midPDiff'] = (OBdf['ask1'] + OBdf['bid1'] - OBdf['ask1_BM'] - OBdf['bid1_BM']) / 2
    HisStd = OBdf['midPDiff'].std()
    HisStdT = HisStd * StdCo  # 乘以一个给定的标准差系数，得到判断阈值
    logger.info(f'"data": "IF UPDATE ORDERBOOK?--Current Adjusted midPrice:{round(midP_RT_Adjust,8)}: LowBound:{round(Fairvalue-HisStdT,8)}--Fairvalue:{round(Fairvalue,8)}--UpBound:{round(Fairvalue+HisStdT,8)}"')
    # 判断条件Result
    if midP_RT_Adjust >= (Fairvalue + HisStdT):
        Result = -1
    elif midP_RT_Adjust <= (Fairvalue - HisStdT):
        Result = 1
    else:
        Result = 0
    return Result


def calcul_orderbook(Fairvalue, spread, OBdf):
    """
    计算orderbook的函数：
    最重要的是控制ask1和bid1的价格，由Fairvalue和spread给出：
    ask1=Fairvalue+1/2*spread
    bid1=Fairvalue-1/2*spread
    最优档的挂单量askS1和bidS1，暂时先由给定的最优档挂单金额(BestPriceOrderNum)计算得到(会加入一点随机量)
    二档以上的价格、挂单量，由随机量给出
    (**)如果中间价波动很大，则spread需要进行放宽处理
    
    orderbook = calcul_orderbook(Fairvalue,spread)
    Input:
    Fairvalue:公平的中间价
    spread:最优买卖价差
    OBdf:处理后的历史orderbook数据(5分钟采样)
    Output:
    orderbook:一个dataframe，有10档价格，4个columns分别为卖单价格(ask1)、卖单量(askS1)、买单价格(bid1)、买单量(bidS1)
    """
    # ----信息预处理----
    ask1_b_10min = OBdf['ask1'][len(OBdf) - 2]
    bid1_b_10min = OBdf['bid1'][len(OBdf) - 2]
    midP_orderbook = Fairvalue
    midP_b_10min = (ask1_b_10min + bid1_b_10min) / 2
    Diff_OB_RT = abs(midP_orderbook / midP_b_10min - 1)  # 更新的orderbook的中间价和10min前的中间价的偏差(%)
    spreadRatio = spread / Fairvalue  # 更新的orderbook的spread比例(spread/midPrice)(%)
    spreadRatio_b_10min = 2 * (ask1_b_10min - bid1_b_10min) / (ask1_b_10min + bid1_b_10min)
    # --计算spread放宽系数--
    # 如果波动偏差大于一个spread比例，我们就认为需要放宽spread(如，spreadRaio=5%，则波动偏差超过5%时，我们要放宽spread来自我保护)
    # 注意：如果10分钟前的(ask1,bid1)数据计算的spreadRatio_b_10min超过了100%，就认为那个时点的挂单出现了错误，则不进行spread放宽调整
    # 随后的orderbook更新，如果没有需要放宽调整的情况，则每次都降低一点，直至1
    global spreadWidenCoef
    if Diff_OB_RT >= spreadRatio and spreadRatio_b_10min <= 100*1e-2:
        x = Diff_OB_RT / spreadRatio  # x = 波动偏差/spread比例
        spreadWidenCoef = max(x, spreadWidenCoef * 0.9)  # 需要调整时，spread放宽系数:x和(0.9*系数前值)中的最大的那一个
    else:
        spreadWidenCoef = spreadWidenCoef * 0.9  # 不需要调整时，spread放宽系数:(0.9*系数前值)，随着orderbook的更新递减，直至为1
        if spreadWidenCoef < 1: 
            spreadWidenCoef = 1
    logger.info(f'"data": "Orderbook Update with spreadWidenCoef({spreadWidenCoef}) -- Diff_OB_RT({Diff_OB_RT}) / spreadRatio{spreadRatio}"')
    # ----ask1、bid1信息----
    # --ask1、bid1价格(依据波动情况进行调整)--
    spread_Adjust = spread * spreadWidenCoef
    ask1 = Fairvalue + 1 / 2 * spread_Adjust
    bid1 = Fairvalue - 1 / 2 * spread_Adjust
    # --ask1、bid1挂单量：服从指数分布，但是数量必须满足一定条件(超出范围重新设值)--
    askS1, bidS1 = 0, 0
    while (askS1 < 0.1 or askS1 > 3 * BestPriceOrderNum or bidS1 < 0.1 or bidS1 > 3 * BestPriceOrderNum):
        askS1 = np.random.exponential(BestPriceOrderNum)
        bidS1 = np.random.exponential(BestPriceOrderNum)
    # ----askn、bidn信息(n>=1)----
    spreadRatio_0_5 = 0.5 * (spread / Fairvalue)
    spreadRatio_1_5 = 1.5 * (spread / Fairvalue)
    delta_ask = list(np.log(np.random.uniform(1 + spreadRatio_0_5, 1 + spreadRatio_1_5, 9)))
    delta_bid = list(np.log(np.random.uniform(1 - spreadRatio_0_5, 1 - spreadRatio_1_5, 9)))
    delta_ask, delta_bid = [np.log(ask1)] + delta_ask, [np.log(bid1)] + delta_bid  # 合并
    askS = list(np.random.uniform(0.1, nonBestPriceOrderNumMax, 9))
    bidS = list(np.random.uniform(0.1, nonBestPriceOrderNumMax, 9))
    askS, bidS = [askS1] + askS, [bidS1] + bidS  # 合并
    # --生成一个orderbook的DataFrame--
    orderbook = pd.DataFrame({'delta_ask': delta_ask, 'askS_ETH': askS, 'delta_bid': delta_bid, 'bidS_ETH': bidS}, index=range(10))
    orderbook['ask'] = np.e ** (orderbook['delta_ask'].cumsum())
    orderbook['askS'] = orderbook['askS_ETH'] / orderbook['ask']
    orderbook['bid'] = np.e ** (orderbook['delta_bid'].cumsum())
    orderbook['bidS'] = orderbook['bidS_ETH'] / orderbook['bid']
    orderbook = orderbook[['ask', 'askS', 'bid', 'bidS']]
    return orderbook


async def create_order(account_manager, Account, BuyorSell, amount, price, expire=86400):
    """
    产生山寨币订单的函数：
    order = await create_order(account_manager,Account,BuyorSell,amount,price)
    根据输入参数情况，向dex输送订单
    (**)优化方向：如何自适应账户数目？
    
    Input:
    Account:账户名称
    BuyorSell:买单还是卖单(只能是Buy或者Sell，注意大小写)
    amount:下单量
    price:下单价格
    Output:
    order:订单类
    """
    try:
        if BuyorSell == 'Buy' or BuyorSell == 'Sell':
            if Account == 'Master':
                if BuyorSell == 'Buy':
                    order = await account_manager.Master.createLimitBuyOrder(symbol, round(amount, 3), round(price, 8), expire=expire)
                else:
                    order = await account_manager.Master.createLimitSellOrder(symbol, round(amount, 3), round(price, 8), expire=expire)
            elif Account == 'Slave1':
                if BuyorSell == 'Buy':
                    order = await account_manager.Slave1.createLimitBuyOrder(symbol, round(amount, 3), round(price, 8), expire=expire)
                else:
                    order = await account_manager.Slave1.createLimitSellOrder(symbol, round(amount, 3), round(price, 8), expire=expire)
            elif Account == 'Slave2':
                if BuyorSell == 'Buy':
                    order = await account_manager.Slave2.createLimitBuyOrder(symbol, round(amount, 3), round(price, 8), expire=expire)
                else:
                    order = await account_manager.Slave2.createLimitSellOrder(symbol, round(amount, 3), round(price, 8), expire=expire)
            elif Account == 'Slave3':
                if BuyorSell == 'Buy':
                    order = await account_manager.Slave3.createLimitBuyOrder(symbol, round(amount, 3), round(price, 8), expire=expire)
                else:
                    order = await account_manager.Slave3.createLimitSellOrder(symbol, round(amount, 3), round(price, 8), expire=expire)
            elif Account == 'Slave4':
                if BuyorSell == 'Buy':
                    order = await account_manager.Slave4.createLimitBuyOrder(symbol, round(amount, 3), round(price, 8), expire=expire)
                else:
                    order = await account_manager.Slave4.createLimitSellOrder(symbol, round(amount, 3), round(price, 8), expire=expire)
            else:
                logger.error(f'"error": "error: def create_order just considers 5 Account"')
        else:
            logger.error(f'"error": "error: BuyorSell must be Buy or Sell"')
        return order
    except:
        traceback.print_exc()
        raise


async def send_order(account_manager, balance, orderbook):
    """
    根据计算的orderbook发送山寨币订单的函数：
    send_order(account_manager,balance,orderbook)
    依据计算的orderbook向dex发送订单簿
    (**)优化方向：如何自适应orderbook的档位(比如，想挂15个档位)？
    
    Input:
    account_manager:账户管理的类
    balance:账户信息的dataframe
    orderbook:之前计算好的想要达到的orderbook
    """
    # 登录账户
    logger.info('"msg": "send_order start"')
    try:
        await account_manager.login()
    except:
        logger.error('"error": "Account Login Timeout!"')
        raise
    # ----判断Master账户能不能提交orderbook----
    balance_Master = balance.reset_index()
    balance_Master = balance_Master[balance_Master['index'] == 'Master']
    ETH_all_Master = balance_Master['cash_total'][0]  # Master的账户中ETH总额
    Inventory_all_Master = balance_Master['Inventory_total'][0]  # Master的账户中山寨币总额
    orderbook_Master = orderbook.copy()
    orderbook_Master['bid_cash'] = orderbook_Master['bid'] * orderbook_Master['bidS']
    bid_all_Master = orderbook_Master['bid_cash'].sum()  # Master要挂出的所有买单之和
    ask_all_Master = orderbook_Master['askS'].sum()  # Master要挂出的所有卖单之和
    if (bid_all_Master >= ETH_all_Master) or (ask_all_Master >= Inventory_all_Master):
        logger.error(f"error: Master Account have No Enough ETH or Inventory! Please Check!")
    # ----开始执行订单操作----
    # --先撤销Master的所有订单--
    NowTimeStamp = f'{int(time.time()*1000)}'
    Master_cancel_all = await account_manager.Master.cancel_all_orders(symbol, NowTimeStamp)
    # --再依据orderbook下单--
    # bid1和ask1先下单下去，其余的orderbook部分一起下
    logger.info('"msg": "place order"')
    global ISOrderbookUpdateNotOK  # 如果没有挂单成功，那么下一轮循环继续挂单
    ISOrderbookUpdateNotOK = 0
    try:
        try:
            bid1 = await create_order(account_manager, 'Master', 'Buy', orderbook['bidS'][0], orderbook['bid'][0])
        except Exception as e:
            logger.error(f'"error": "place order bid1 {bid1} error :[{type(e)}]  {e}"')
            raise
        else:
            logger.info(f'"msg": "place order bid1 {bid1}"')
        try:
            ask1 = await create_order(account_manager, 'Master', 'Sell', orderbook['askS'][0], orderbook['ask'][0])
        except Exception as e:
            logger.error(f'"error": "place order ask1 {ask1} error :[{type(e)}]  {e}"')
            raise
        else:
            logger.info(f'"msg": "place order ask1 {ask1}"')
        bid2 = await create_order(account_manager, 'Master', 'Buy', orderbook['bidS'][1], orderbook['bid'][1])
        logger.info(f""""msg": "place order bid2: orderId is {bid2['order']['orderId']}" """)
        ask2 = await create_order(account_manager, 'Master', 'Sell', orderbook['askS'][1], orderbook['ask'][1])
        logger.info(f""""msg": "place order ask2: orderId is {ask2['order']['orderId']}" """)
        bid3 = await create_order(account_manager, 'Master', 'Buy', orderbook['bidS'][2], orderbook['bid'][2])
        logger.info(f""""msg": "place order bid3: orderId is {bid3['order']['orderId']}" """)
        ask3 = await create_order(account_manager, 'Master', 'Sell', orderbook['askS'][2], orderbook['ask'][2])
        logger.info(f""""msg": "place order ask3: orderId is {ask3['order']['orderId']}" """)
        bid4 = await create_order(account_manager, 'Master', 'Buy', orderbook['bidS'][3], orderbook['bid'][3])
        logger.info(f""""msg": "place order bid4: orderId is {bid4['order']['orderId']}" """)
        ask4 = await create_order(account_manager, 'Master', 'Sell', orderbook['askS'][3], orderbook['ask'][3])
        logger.info(f""""msg": "place order ask4: orderId is {ask4['order']['orderId']}" """)
        bid5 = await create_order(account_manager, 'Master', 'Buy', orderbook['bidS'][4], orderbook['bid'][4])
        logger.info(f""""msg": "place order bid5: orderId is {bid5['order']['orderId']}" """)
        ask5 = await create_order(account_manager, 'Master', 'Sell', orderbook['askS'][4], orderbook['ask'][4])
        logger.info(f""""msg": "place order ask5: orderId is {ask5['order']['orderId']}" """)
        bid6 = await create_order(account_manager, 'Master', 'Buy', orderbook['bidS'][5], orderbook['bid'][5])
        logger.info(f""""msg": "place order bid6: orderId is {bid6['order']['orderId']}" """)
        ask6 = await create_order(account_manager, 'Master', 'Sell', orderbook['askS'][5], orderbook['ask'][5])
        logger.info(f""""msg": "place order ask6: orderId is {ask6['order']['orderId']}" """)
        bid7 = await create_order(account_manager, 'Master', 'Buy', orderbook['bidS'][6], orderbook['bid'][6])
        logger.info(f""""msg": "place order bid7: orderId is {bid7['order']['orderId']}" """)
        ask7 = await create_order(account_manager, 'Master', 'Sell', orderbook['askS'][6], orderbook['ask'][6])
        logger.info(f""""msg": "place order ask7: orderId is {ask7['order']['orderId']}" """)
        bid8 = await create_order(account_manager, 'Master', 'Buy', orderbook['bidS'][7], orderbook['bid'][7])
        logger.info(f""""msg": "place order bid8: orderId is {bid8['order']['orderId']}" """)
        ask8 = await create_order(account_manager, 'Master', 'Sell', orderbook['askS'][7], orderbook['ask'][7])
        logger.info(f""""msg": "place order ask8: orderId is {ask8['order']['orderId']}" """)
    except Exception as e:
        ISOrderbookUpdateNotOK = 1
        logger.info('"msg": "--Warning: Orderbook Update Not Scucess"')
    # --记录一下Master自己挂出来的bid1和ask1，更新到一个缓存数据结构BestPrice_Master--
    global BestPrice_Master
    ask1_Master = round(orderbook['ask'][0], 8)
    bid1_Master = round(orderbook['bid'][0], 8)
    BestPrice_Master.append([ask1_Master, bid1_Master])
    midP_Master = round((ask1_Master + bid1_Master) / 2, 8)
    logger.info(f'"data": "Master place ask1({ask1_Master}) and bid1({bid1_Master}), midP_Master is {midP_Master}"')
    # --最后更新一下orderbookUpdateCount--
    # (注意)这里放在最后，是因为必须有了orderbook下单才算真的做完了一轮
    global orderbookUpdateCount
    orderbookUpdateCount = orderbookUpdateCount + 5  # 一旦orderbook出现了Update，则加快自成交(让orderbookUpdateCount增加)


async def SelfTrade(account_manager, balance, Fairvalue, spread, OB_RealTime):
    """
    处理自成交的函数：
    SelfTrade(account_manager, balance, Fairvalue, spread, OB_RealTime)
    调用这个函数，就会完成一次自成交操作
    
    Input:
    account_manager:账户管理的类
    balance:账户信息的dataframe
    Fairvalue:之前计算好的公平的中间价
    spread:之前计算好的最优买卖价差
    OB_RealTime:当前实时的ask1/bid1数据
    """
    # ----登录账户----
    try:
        await account_manager.login()
    except:
        logger.error('"error": "Account Login Timeout!"')
        raise
    # ----信息预处理----
    IfSelfTrade = 1  # 是否自成交的判断变量
    # 首先做一个判断，如果Fairvalue与当前的中间价差距超过max(2%, 2*spreadRatio_orderbook)，则强制不产生自成交
    ask1_RT = OB_RealTime['ask1'][0]
    bid1_RT = OB_RealTime['bid1'][0]
    midP_orderbook = Fairvalue
    midP_RT = (ask1_RT + bid1_RT) / 2
    spreadRatio_orderbook = spread / Fairvalue
    if abs(midP_orderbook / midP_RT - 1) >= max(2*1e-2, 2*spreadRatio_orderbook):
        IfSelfTrade = 0  # 将IfSelfTrade设为0，即强制不进行自成交
        logger.info(f'"data": "No SelfTrade, Because Gap Too Large -- Market RealTime midP: {midP_RT} -- Orderbook midP: {midP_orderbook}"')
    # balance1，所有slave账户按照库存总额排序，以此确定用哪个slave来成交
    balance1 = balance.sort_values('Inventory_total').reset_index()
    balance1 = balance1[balance1['index'] != 'Master']
    Slave_Cash_Total = balance1['cash_total'].sum()
    Slave_Inventory_Total = balance1['Inventory_total'].sum() * midP_orderbook
    Master_Cash_free = balance[balance.index == 'Master']['cash_free'][0]
    Master_Inventory_free = balance[balance.index == 'Master']['Inventory_free'][0] * midP_orderbook
    # 以slave账户总库存来确定自成交的方向--slave账户总山寨币库存低则slave买入，总现金库存低则slave卖出，其余情况随机
    # 以master账户可用来确定自成交方向--master账户山寨币可用低则slave卖出，现金可用低则slave买入，其余情况随机
    if (Slave_Cash_Total < Slave_Cash_Total_MIN) or (Master_Inventory_free < Master_Inventory_free_MIN):
        Direction = -1
    elif (Slave_Inventory_Total < Slave_Inventory_Total_MIN) or (Master_Cash_free < Master_Cash_free_MIN):
        Direction = 1
    else:
        Direction = int(round(np.random.uniform(0, 1) ,0)) * 2 - 1
    logger.info(f'"data": "SelfTrade Direction: {Direction}"')
    logger.info(f'"data": "-- Slave_Cash_Total: {Slave_Cash_Total} -- Slave_Inventory_Total: {Slave_Inventory_Total}"')
    logger.info(f'"data": "-- Master_Cash_free: {Master_Cash_free} -- Master_Inventory_free: {Master_Inventory_free}"')
    # ----开始自成交操作(在IfSelfTrade==1的时候)----
    if IfSelfTrade == 1:
        # --首先，计算自成交的价格、量等信息--
        global InventoryWant2be
        global BestPrice_Master
        # 随机的交易量，满足条件：在区间(0.11, SelfTradeTotalMax)内
        SelfOrderTotal = 0
        while (SelfOrderTotal < 0.11 or SelfOrderTotal > SelfTradeTotalMax):
            SelfOrderTotal = np.random.exponential(SelfTradeTotalMax / 2)
        # 自成交单子的价格，靠近中间价，谨防被套利
        SelfBuyPrice = ask1_RT - 1 / 2 * np.random.uniform(0.95, 1.05) * (ask1_RT - bid1_RT)  # 自成交的买单为当前的中间价附近
        SelfSellPrice = bid1_RT + 1 / 2 * np.random.uniform(0.95, 1.05) * (ask1_RT - bid1_RT)  # 自成交的卖单为当前中间价附近
        # --然后，发出自成交的单子，并更新InvestoryWant2be--
        randNum = np.random.uniform(0, 1)  # 随机变量来确定交易量是否取整
        if Direction == 1:
            AccountName = balance1.iloc[0]['index']  # 库存最小的slave执行买入
            SelfOrderType = 'BuyOrder'
            SelfOrderPrice = SelfBuyPrice
            if randNum < 1 / 2:  # 依据一个随机量来确定:(1)下单量取整；(2)不取整
                SelfVolume = SelfOrderTotal / SelfOrderPrice
                if SelfVolume >= 10:
                    SelfVolume = round(SelfVolume, 0) + 1
                else:
                    SelfVolume = round(SelfVolume, 1) + 0.1
            else:
                SelfVolume = SelfOrderTotal / SelfOrderPrice
            SelfOrder1 = await create_order(account_manager, 'Master', 'Sell', SelfVolume, SelfOrderPrice, 5)
            SelfOrder2 = await create_order(account_manager, AccountName, 'Buy', SelfVolume, SelfOrderPrice, 1)
            InventoryWant2be = InventoryWant2be - SelfVolume * TakerOrderCost  # 山寨币买方为吃单，用Taker费率计算交易费用，再更新合宜库存量
        elif Direction == -1:
            AccountName = balance1.iloc[-1]['index']  # 库存最大的slave执行卖出
            SelfOrderType = 'SellOrder'
            SelfOrderPrice = SelfSellPrice
            if randNum < 1 / 2:  # 依据一个随机量来确定:(1)下单量取整；(2)不取整
                SelfVolume = round(SelfOrderTotal / SelfOrderPrice, 0) + 1
            else:
                SelfVolume = SelfOrderTotal / SelfOrderPrice
            SelfOrder1 = await create_order(account_manager, 'Master', 'Buy', SelfVolume, SelfOrderPrice, 5)
            SelfOrder2 = await create_order(account_manager, AccountName, 'Sell', SelfVolume, SelfOrderPrice, 1)
            InventoryWant2be = InventoryWant2be - SelfVolume * MakerOrderCost  # 山寨币买方为挂单，用Maker费率计算交易费用，再更新合宜库存量
        logger.info(f'"data": "--SelfTradeOccurs: {AccountName} send a {SelfOrderType} at {SelfOrderPrice} with (amount,total):({SelfVolume},{SelfOrderTotal}), Update InventoryWant2be({InventoryWant2be})--"')


def data_append(msg, msg_BM):
    """
    向一个Cache类的实例append最新数据：
    将经过整理的实时数据，每5分钟append到一个Cache实例，这个实例缓存了历史5分钟orderbook采样数据
    
    data_append(msg,msg_BM,cache)
    Input:
    msg:dex的实时数据
    msg_BM:bemchmark交易所的实时数据
    """
    # ----当前的实时数据的处理----
    # --benchmark数据--
    ask1_BM = msg_BM.get("asks")[0][0]
    bid1_BM = msg_BM.get("bids")[0][0]
    # --dex数据--
    ask1 = msg.get("asks")[0][0]
    bid1 = msg.get("bids")[0][0]
    # --合并生成一个list--
    appendList = [ask1, bid1, ask1_BM, bid1_BM]
    # ----满足条件时append数据----
    x = datetime.datetime.utcfromtimestamp(msg.get("nonce"))
    y = x.strftime("%Y-%m-%d %H:%M:%S")
    UTCordertime = datetime.datetime.strptime(y, "%Y-%m-%d %H:%M:%S")
    if UTCordertime.minute % 5 == 0 and UTCordertime.second < 30:
        cache.append(appendList)


# 读到dex和benchmark交易所的数据后的函数
async def deal_with_data(msg, msg_BM, BestPrice_Master, cache, account_manager, loop):
    logger.info('"msg": "======start get and manufacture data======"')
    # ----当前的实时数据的处理----
    func1 = partial(data_transform_RealTime, msg, msg_BM)
    OB_RealTime = await loop.run_in_executor(None, func1)  # 计算部分放到多进程执行器中执行
    # ----历史数据的处理----
    func2 = partial(data_transform_His, cache)
    OBdf = await loop.run_in_executor(None, func2)
    # ----做市商账户信息----
    balance = await calcul_balance(account_manager)
    logger.info(f'"data": "get balance {balance}"')
    # ----计算spread----
    logger.info('"msg": "======start calculate Key Data======"')
    func3 = partial(calcul_spread, OBdf, OB_RealTime)
    spread = await loop.run_in_executor(None, func3)
    # ----计算Fairvalue----
    func4 = partial(calcul_fairvalue, balance, InventoryWant2be, OBdf, OB_RealTime)
    Fairvalue = await loop.run_in_executor(None, func4)
    # ----计算判断条件Result----
    func5 = partial(calcul_result, OBdf, OB_RealTime, BestPrice_Master, Fairvalue)
    Result = await loop.run_in_executor(None, func5)
    # ----根据更新条件和orderbook更新是否正常的状态，判断是否需要更新orderbook----
    global ISOrderbookUpdateNotOK
    if (Result != 0 or ISOrderbookUpdateNotOK == 1):  # 满足更新条件或orderbook更新不正常时
        logger.info('"msg": "====orderbook upate===="')
        # ----依据Result结果，计算orderbook，并发送订单----
        func6 = partial(calcul_orderbook, Fairvalue, spread, OB_RealTime)
        orderbook = await loop.run_in_executor(None, func6)
        logger.info('"msg": "orderbook done!"')
        # ----发送订单----
        await send_order(account_manager, balance, orderbook)
        logger.info('"msg": "send_order done!"')
        logger.info(f'"data": "***orderbook is: {str(orderbook)}***"')
    else:
        logger.info('"msg": "====no orderbook update===="')
        # ----不更新订单簿的时候，进行自成交相关操作----
        global orderbookUpdateCount
        # 不自成交时，orderUpdateCount随机加0或1
        if orderbookUpdateCount < SelfTradeFreq:
            orderbookUpdateCount += int((round(np.random.uniform(0, 1))))
            logger.info(f'"data": "--No SelfTradeOccurs: After About {SelfTradeFreq - orderbookUpdateCount + 0.5}min will Occour!"')
        # 自成交时，orderUodateCount变回0
        else:
            await SelfTrade(account_manager, balance, Fairvalue, spread, OB_RealTime)
            orderbookUpdateCount = 0
        
    # ----向缓存append数据----
    data_append(msg, msg_BM)


# ------(2)main函数------
async def main():
    # 定义loop
    loop = asyncio.get_event_loop()
    # 登录账户
    logger.info('"msg": "init account"')
    all_account = [Account(clz=dex, **i) for i in ACCOUNTS]  # 注意，这里的登录方法用的是MMmarket名，需要检查是否一致(目前dex和dex_test是一致的)
    logger.info('"msg": "init account manager"')
    account_manager = AccountManager(all_account)
    # 起一个协程来接收benchmark的数据
    BM_ch = CurrentData("depth", benchmark, timeout=3, symbol=symbol_BM)
    BM2_ch = CurrentData("depth", benchmark2, timeout=3, symbol=symbol_BM)
    # benchmark频道的错误计数
    benchmarkDataCount = 0
    # 由推送数据驱动
    logger.info('"msg": "init sub"')
    async with Subscribe("depth", MMmarket, symbol) as MM_ch:
        logger.info('"msg": "#################################This Round Start#################################"')
        async for msg in MM_ch:
            try:
                logger.info(f'"data": "Latest raw data {cache._content[-1]}"')
                # 如果benchmark的timeout错误连续出现了benmarkDataCountMax次，则转而读取备用benchmark频道
                if benchmarkDataCount < benmarkDataCountMax:
                    logger.info(f'"msg": "Try benchmark channel: {benchmarkDataCount+1} times"')
                    msg_BM = BM_ch.value
                    benchmarkDataCount = 0  # 从benchmark频道中成功读数就重置为0
                else:
                    logger.error(f'"error": "There is someting wrong with benchmark channel, Try benchmark2 channel: {benchmarkDataCount-3} times, Please Check Channel ({benchmark})!"')
                    msg_BM = BM2_ch.value
                    benchmarkDataCount = 0  # 从benchmark备用频道中成功读数就重置为0
            except Exception as e:
                benchmarkDataCount = benchmarkDataCount + 1
                print(benchmarkDataCount)
                logger.info(f'"error": "Get Benchmark Data error {str(e)}"')
                continue

            asyncio.ensure_future(deal_with_data(msg, msg_BM, BestPrice_Master, cache, account_manager, loop))

            # break


# logger.setLevel(logging.DEBUG)
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())


"""
# 手动向市场下单
async def send_order():
    
    all_account = [Account(clz=dex, **i) for i in ACCOUNTS]
    account_manager = AccountManager(all_account)
    
    # 登录账户并读取数据
    await account_manager.login()
    # ----读账户操作----
    # ----开始执行订单操作----
#    global bid1
#    bid1 = await create_order(account_manager, "Master", 'Buy', orderbook['bidS'][0], orderbook['bid'][0])
#    global bid2
#    bid2 = await create_order(account_manager, "Master", 'Buy', 10000, 0.00002000)
#    global Cancel
#    NowTimeStamp = f'{int(time.time()*1000)}'
#    Cancel = await account_manager.Master.cancel_all_orders('VITE/ETH',NowTimeStamp)


loop = asyncio.get_event_loop()
loop.run_until_complete(send_order())


async def calcul_balance():
    
    all_account = [Account(clz=dex, **i) for i in ACCOUNTS]
    account_manager = AccountManager(all_account)
    
    # 登录账户并读取数据
    await account_manager.login()
    total_balance = await account_manager.total_balance()
    
    InventorySymbol = symbol.split('/')[0]  # 山寨币的symbol
    
    global balance
    
    balance = pd.DataFrame(total_balance).T
    balance['cash_total'] = balance['ETH'].apply(lambda x: float(x['total']))
    balance['cash_free'] = balance['ETH'].apply(lambda x: float(x['free']))
    balance['Inventory_total'] = balance[InventorySymbol].apply(lambda x: float(x['total']))
    balance['Inventory_free'] = balance[InventorySymbol].apply(lambda x: float(x['free']))
    balance = balance[['cash_total', 'cash_free', 'Inventory_total', 'Inventory_free']]
    print(balance)
    
loop = asyncio.get_event_loop()
loop.run_until_complete(calcul_balance())
"""
