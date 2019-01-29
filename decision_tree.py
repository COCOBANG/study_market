# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:25:47 2019

@author: Administrator
"""

import pandas as pd
import numpy as np

data = {
'id': list(range(1,18)), 
'Seze': ['Qing', 'Wu', 'Wu', 'Qing', 'Qian', 'Qing', 'Wu', 'Wu', 'Wu', 'Qing', 'Qian', 'Qian', 'Qing', 'Qian', 'Wu', 'Qian', 'Qing'],
'Gendi': ['Quan', 'Quan', 'Quan', 'Quan', 'Quan', 'Shao', 'Shao', 'Shao', 'Shao', 'Ying', 'Ying', 'Quan', 'Shao', 'Shao', 'Shao', 'Quan', 'Quan'],
'Qiaosheng': ['Zhuo', 'Chen', 'Zhuo', 'Chen', 'Zhuo', 'Zhuo', 'Zhuo', 'Zhuo', 'Chen', 'QingC', 'QingC', 'Zhuo', 'Zhuo', 'Chen', 'Zhuo', 'Zhuo', 'Chen'],
'Wenli': ['QingX', 'QingX', 'QingX', 'QingX', 'QingX', 'QingX', 'ShaoH', 'QingX', 'ShaoH', 'QingX', 'Mo', 'Mo', 'ShaoH', 'ShaoH', 'QingX', 'Mo', 'ShaoH'],
'Qibu': ['Ao', 'Ao', 'Ao', 'Ao', 'Ao', 'ShaoA', 'ShaoA', 'ShaoA', 'ShaoA', 'Ping', 'Ping', 'Ping', 'Ao', 'Ao', 'ShaoA', 'Ping', 'ShaoA' ],
'Chugan': ['YingH', 'YingH', 'YingH', 'YingH', 'YingH', 'Ruan', 'Ruan', 'YingH', 'YingH', 'Ruan', 'YingH', 'Ruan','YingH', 'YingH', 'Ruan','YingH', 'YingH'],
'Haogua': ['Hao']*8+['Huai']*9
}

xg = pd.DataFrame(data)


def Ent_cal(df):
    '''
    计算信息熵
    '''
    E = df[label_name].value_counts().values
    E = E / E.sum()
    E = - (E * np.log2(E)).sum()
    return E


def Ent_aft_cal(df, prop):
    '''
    计算按属性分组后的信息熵
    Inputs:
        df: 分组前的数据表df
        prop: 分组变量名
    '''
    dfgp = df.groupby(prop)
    df_propvallen = len(df[prop].value_counts())  # 该属性有多少个取值
    df_len = len(df)  # 总共有多少观测
    E_arr = np.zeros(df_propvallen)  # 每个分组的信息熵初始化
    ratio_arr = np.zeros(df_propvallen)  # 每个分组占比初始化
    i = 0  # 位置信息初始化
    for gp in dfgp:
        E_arr[i] = Ent_cal(gp[1])  # gp[1]为groupby的每一个df
        ratio_arr[i] = len(gp[1]) / df_len
        i += 1
    return sum(E_arr * ratio_arr)


def TreeGenerate(df, df_start, parent=None):
    '''
    计算决策树
    Inputs:
        df: 这一层递归所依赖的DataFrame
        df_start: 整个树的起点df，这是为了能够取到属性的全部取值组合(由于通过不断地按条件取值，df可能已经不包含某些属性的某些取值了)
        parent: 父节点信息，以"{父节点特征}=={父节点特征取值}"为格式
    '''
    
    global node, decision_tree  # 全局变量
    
    # (1) 递归结束条件判断：节点的训练子集满足一些条件时
    df_clslen = len(df[label_name].value_counts())  # 该节点的训练子集有多少个不同的分类标签
    columns = [i for i in df.columns if i not in ('id', label_name)]
    df_proplen = len(columns)  # 该节点的训练子集有多少个属性
    if df_clslen < 2:  # 节点只有一种分类标签时
        decision_tree.update({node: {'parent': parent, 'property': None, 'label': df[label_name][0]}})
        node += 1
    elif df_proplen < 2:  # 节点只有一个属性时，取训练子集里类别标记最多的类别
       decision_tree.update({node: {'parent': parent, 'property': None, 'label': df[label_name].value_counts().index[0]}})
       node += 1
      # (2) 节点有多种分类标签且多个属性时，递归
    else:
        # (2.1) 找到该节点的"信息增益"最大属性
        Gain = {}
        for prop in columns:
            Gain.update({prop: Ent_cal(df) - Ent_aft_cal(df, prop)})
        prop_max = max(Gain, key=Gain.get)  # 信息增益最大的属性prop_max
        # (2.2) 本节点完成，update决策树上该节点信息
        decision_tree.update({node: {'parent': parent, 'property': prop_max, 'label': None}})
        node += 1        
        # (2.3) 递归生成下一个节点
        columns_next = [i for i in df.columns if i != prop_max]  # 下一个节点不再考虑本节点的属性
        for prop_value in xg[prop_max].value_counts().index:  # 注意这里用df_start来遍历(因为df可能会因为某些取值组合观测为0而忽略了一些取值的可能)
            # 生成下一个节点的训练子集：属性取值为prop_max之一，且除去该属性的其它观测
            df1 = df[df[prop_max] == prop_value][columns_next].reset_index(drop=True)
            if len(df1) == 0:  # 没有观测时就用父节点类别标记最多的标记(但是parent信息是该子节点信息)
                decision_tree.update({node: {'parent': f'{prop_max}=={prop_value}', 'property': None, 'label': df[label_name].value_counts().index[0]}})
                node += 1
            else:  # 有观测时就调用函数进入递归
                TreeGenerate(df1, xg, parent=f'{prop_max}=={prop_value}')


### 测试
label_name = 'Haogua'           
decision_tree ={}
# {'node': {'parent': , 'property': , 'label': }}
## 叶节点情况：parent/property/label都取值
## 根节点情况：parent/label不取值，property取值
## 中间节点情况：parent/property取值，label不取值 
node = 0

TreeGenerate(xg, xg)