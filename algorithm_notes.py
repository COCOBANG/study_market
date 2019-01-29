# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 09:23:12 2019

@author: Administrator
"""

import numpy as np

# 二分查找
### 找到即可，有重复值时找重复值的位置和length有关
def binary_search(list_input, item):
    low = 0
    high = len(list_input) - 1
    while low <= high:
        mid = (low+high) // 2
        guess = list_input[mid]
        if guess == item:
            return mid
        elif guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None

### 如果出现重复值，则找第一个值出现的位置(first)
def binary_search_first(list_input, item):
    low = 0
    high = len(list_input) - 1
    while low <= high:
        mid = (low+high) // 2
        guess = list_input[mid]
        if guess < item:
            low = mid + 1
        else:  # 和标准二分查找的不同之处:(1)即使guss=item也不一定停止搜索; (2)high的更新不是(mid-1)
            high = mid
        if low == high:
            return low
    return None

### 如果出现重复值，则找最后一个值出现的位置(last)
def binary_search_last(list_input, item):
    low = 0
    high = len(list_input) - 1
    while low <= high:
        mid = (low+high) // 2 + 1  # 注意mid和之前的不同
        guess = list_input[mid]
        if guess > item:
            high = mid - 1
        else:  # 和标准二分查找的不同之处:(1)即使guss=item也不一定停止搜索; (2)low的更新不是(mid+1)
            low = mid
        if low == high:
            return low
    return None

list_input = [1, 5, 9, 9, 12]
print('test1 of binary_search:')
print(f'Inputs: {list_input}')
print(f'只要求找到即可的binary_search: {binary_search(list_input, 9)}')
print(f'如果有重复值，则找到的是第一个出现的位置: {binary_search_first(list_input, 9)}')
print(f'如果有重复值，则找到的是最后一个出现的位置: {binary_search_last(list_input, 9)}')

list_input = [1, 5, 9, 9, 9, 9, 12]
print('test2 of binary_search:')
print(f'Inputs: {list_input}')
print(f'只要求找到即可的binary_search: {binary_search(list_input, 9)}')
print(f'如果有重复值，则找到的是第一个出现的位置: {binary_search_first(list_input, 9)}')
print(f'如果有重复值，则找到的是最后一个出现的位置: {binary_search_last(list_input, 9)}')


# 选择排序
def findSmallest(arr):
    smallest = arr[0]
    smallest_index = 0
    for i in range(1, len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index

def selectionSort(arr):
    newArr = []
    for i in range(len(arr)):
        smallest = findSmallest(arr)
        newArr.append(arr.pop(smallest))  # list.pop，制定索引删除(list.remove是制定元素删除)，返回被删除的元素
    return newArr

list_input = [14, 5, 7, 1, 12]
print(selectionSort(list_input))


# 快速排序
def quicksort(array):
    if len(array) < 2:  # 基线条件
        return array
    else:
        pivot = array[0]
        less = [i for i in array[1:] if i <= pivot]
        greater = [i for i in array[1:] if i > pivot]
    
    return quicksort(less) + [pivot] + quicksort(greater)  # 递归

list_input = [14, 5, 7, 1, 12]
print(quicksort(list_input))


# 归并排序
def mergesort(seq):
    mid = len(seq) // 2
    lft, rgt = seq[:mid], seq[mid:]
    if len(lft) > 1:  # 基线条件
        lft = mergesort(lft)
    if len(rgt) > 1:
        rgt = mergesort(rgt)
    res = []  # 递归后处理--将lft和rgt的结果整合在res里(逐一将最大值加入res--lft和rgt的最大值肯定在末尾)
    while lft and rgt:  # while循环--逐一将最大值加入res，直至lft或rgt成为空的list
        if lft[-1] >= rgt[-1]:
            res.append(lft.pop())
        else:
            res.append(rgt.pop())
    res.reverse()
    return (lft or rgt) + res

list_input = [14, 5, 7, 1, 12]
print(mergesort(list_input))


# 图与广度优先搜索
## 利用字典构建一个图
graph = {}
graph["you"] = ["alice", "bob", "claire"]
graph["bob"] = ["anuj", "peggy"]
graph["alice"] = ["peggy"]
graph["claire"] = ["thom", "jonny"]
graph["anuj"] = []
graph["peggy"] = []
graph["thom"] = []
graph["jonny"] = []

# 广度优先搜索
from collections import deque

def search(name):
    search_queue = deque()
    search_queue += graph[name]
    searched = []  # 记录已经检查过的键
    while search_queue:  # 直到queue不再有值(图的所有节点都搜索完了)
        person = search_queue.popleft()
        if person not in searched:
            if person_is_seller(person):
                print(person + " is a mango seller!")
                return True
            else:
                #search_queue += graph[person]  # 没有搜索结果时，把这个person的下一步节点添加到queue里面
                search_queue.extend(graph[person])  # 用标准类的方法extend一般会比通用算符(+=)更有效率一点
                searched.append(person)
    return False

def person_is_seller(name):  # 判断是否是芒果商的函数(简单处理:看名字最后一个字是否为m)
    return name[-1] == 'm'

# test
search('you')


# 加权图和狄克斯特拉算法(练习7.1)
# 利用3个字典来描述加权图求解过程
## (1)graph:加权图(包括和邻居的节点和权重信息)
graph = {}
graph["start"] = {}  # 字典的嵌套
graph["start"]["a"] = 5
graph["start"]["b"] = 2
graph["a"] = {}  # 字典的嵌套
graph["a"]["c"] = 4
graph["a"]["d"] = 2
graph["b"] = {}  # 字典的嵌套
graph["b"]["a"] = 8
graph["b"]["d"] = 7
graph["c"] = {}  # 字典的嵌套
graph["c"]["d"] = 6
graph["c"]["fin"] = 3
graph["d"] = {}  # 字典的嵌套
graph["d"]["fin"] = 1
graph["fin"] = {}  # 终点没有邻居
print("图的节点数：" + f"{len(graph)}")
print("图的边数：" + f"{sum([len(i) for i in graph.values()])}")
## (2)costs:节点cost信息，会不断更新
infinity = float('inf')
costs = {}
costs["a"] = 5  # 从起点开始的节点的初始值(其余节点--包括终点--都先设置为+inf)
costs["b"] = 2
costs["c"] = infinity
costs["d"] = infinity
costs["fin"] = infinity
## (3)parents:父节点信息,本质上是路径信息，会不断更新
parents = {}
parents["a"] = "start"  #从起点开始的节点的初始值(其余节点--包括终点--都先设置为None)
parents["b"] = "start"
parents["c"] = None
parents["d"] = None
parents["fin"] = None

# find_lowest_cost_node
def find_lowest_cost_node(costs):
    # 最低开销节点的初始化(开销值和节点)
    lowest_cost = float("inf")
    lowest_cost_node = None
    # 开始在所有节点中查找
    for node in costs.keys():  # node实际上是costs的key
        cost = costs[node]
        if (cost < lowest_cost) and (node not in processed):  # 注意后一个条件，processed实际上是一个全局变量，处理过的节点就不再处理了
            lowest_cost = cost
            lowest_cost_node = node
    return lowest_cost_node

# 开始狄克斯特拉算法
## 首先，初始化相关变量
processed = []  # 已处理节点list，注意，在这里processed是一个全局变量
node = find_lowest_cost_node(costs)  # 找到初始的cost最低的节点(由于costs的初始值只和start的邻居有关，所以这里找到的肯定是start邻居里cost最低的节点)
## 然后，不断在最低开销节点上计算它到它的邻居的成本，如果发现这些成本比原来记录的要低，则更新costs(new_cost)和parents(该最低开销节点)
while node is not None:  # 找不到cost最低的节点时结束(或者说所有节点都被处理完)
    cost = costs[node]
    neighbors = graph[node]
    for n in neighbors.keys():  # (a)在最低开销节点上讨论它到它各个邻居的成本
        new_cost = cost + neighbors[n]  # 到邻居节点的开销，是当前节点的开销(已经是最低了)加上当前节点去邻居节点的开销
        if costs[n] > new_cost:
            costs[n] = new_cost  # 更新为新的开销值
            parents[n] = node  # 父节点就是当前讨论的最低节点
    processed.append(node)      # (b)讨论结束后将这个节点append到processed里去，避免以后再来讨论这个节点
    node = find_lowest_cost_node(costs)  # (c)再找新的最低开销节点，直到返回None



    