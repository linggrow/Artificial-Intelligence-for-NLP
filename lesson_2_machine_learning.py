import random
import math
import time
from math import radians, cos, sin, asin, sqrt, acos
import pandas as pd
import numpy as np
import re
from collections import Counter
import jieba
from functools import reduce
from operator import add, mul
import matplotlib.pyplot as plt
# % matplotlib inline
import matplotlib
import networkx as nx
from collections import defaultdict
import requests
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.datasets import load_boston
from geopy import distance
from geopy.geocoders import Nominatim
from icecream import ic

# print(matplotlib.__path__)
# import requests
# url = 'https://movie.douban.com/subject/26931786/?from=showing'
# response = requests.get(url)

from sklearn.datasets import load_boston
data = load_boston()

X, y = data['data'], data['target']


def draw_rm_and_price():
    plt.scatter(X[:, 5], y)

draw_rm_and_price()

def price(rm, k, b):
    """f(x) = k * x + b"""
    return k * rm + b


# 画出初始随借
X_rm = X[:, 5]
k = random.randint(-100, 100)
b = random.randint(-100, 100)
price_by_random_k_and_b = [price(r, k, b) for r in X_rm]

draw_rm_and_price()
plt.scatter(X_rm, price_by_random_k_and_b)

def loss(y, y_hat):
    return sum((y_i - y_hat_i) **2 for y_i,y_hat_i in zip(list(y), list(y_hat))) / len(list(y))

def loss_mse(y, y_hat):
    return sum((y_i - y_hat_i) **2 for y_i,y_hat_i in zip(list(y), list(y_hat))) / len(list(y))


def loss_mae(y, y_hat):
    return sum(np.abs(y_i - y_hat_i) for y_i,y_hat_i in zip(list(y), list(y_hat))) / len(list(y))



trying_times = 2000
min_loss = float('inf')

best_k, best_b = None, None

for i in range(trying_times):
    k = random.random() * 200 - 100
    b = random.random() * 200 - 100
    price_by_random_k_and_b = [price(r, k, b) for r in X_rm]

    current_loss = loss(y, price_by_random_k_and_b)
    a = 0

    if current_loss < min_loss:
        min_loss = current_loss
        best_k, best_b = k, b
        print('When time is : {}, get best_k:{} best_b:{}, and the loss is:{}'.format(
            i, best_k, best_k, min_loss
        ))

X_rm = X[:, 5]
k = 15
b = -68
price_by_random_k_and_b = [price(r, k, b) for r in X_rm]

# draw_rm_and_price()
# plt.scatter(X_rm, price_by_random_k_and_b)
# plt.show()

trying_times = 1000
min_loss = float('inf')

best_k = random.random() * 200 - 100
best_b = random.random() * 200 - 100

direction = [
    (+1, -1),  # first element: k's change direction, second element: b's change direction
    (+1, +1),
    (-1, -1),
    (-1, +1),
]

next_direction = random.choice(direction)

scalar = 0.1
update_time = 0

for i in range(trying_times):
    k_direction, b_direction = next_direction
    current_k, current_b = best_k + k_direction * scalar, best_b + b_direction * scalar

    price_by_k_and_b = [price(r, current_k, current_b) for r in X_rm]
    current_loss = loss(y, price_by_k_and_b)
    a = 0

    if current_loss < min_loss:
        a = 0
        min_loss = current_loss
        best_k, best_b = current_k, current_b

        next_direction = next_direction
        update_time += 1

        if update_time % 10 == 0:
            print('When time is : {}, get best_k : {} besk_b:{}, and the loss is:{}'.format(i, best_k, best_b,min_loss))
    else:
        next_direction = random.choice(direction)

#
# 找对改变的方向
# 让他变化，监督学习

def partial_k(x, y, y_hat):
    n = len(y)
    gradient = 0

    for x_i, y_i, y_hat_i in zip(list(x), list(y), list(y_hat)):
        gradient += (y_i - y_hat_i) * x_i
    return -2 / n * gradient

def partial_b(x, y, y_hat):
    n = len(y)
    gradient = 0

    for y_i, y_hat_i, in zip(list(y), list(y_hat)):
        gradient += (y_i - y_hat_i)

    return -2 / n*gradient


trying_times = 2000
X, y = data["data"], data["target"]
min_loss = float('inf')

current_k = random.random() * 200 - 50
current_b = random.random() * 200 - 100
learning_rate = 1e-04

update_time = 0
tmp = 0
for i in range(trying_times):
    price_by_k_and_b = [price(r, current_k, current_b) for r in X_rm]

    current_loss = loss(y, price_by_k_and_b)
    # current_loss = loss_mae(y, price_by_k_and_b)
    # current_loss = loss_mse(y, price_by_k_and_b)

    if current_loss < min_loss:
        min_loss = current_loss
        best_k, best_b = current_k, current_b

        if i % 50 == 0:
            print('When time is : {}, get best_k: {} best_b: {}, and the loss is: {}'.
                  format(i, best_k, best_b, min_loss))
            plt.clf()
            # plt.close()  # 清空窗口，再次打开的时候窗口会改变位置
            plt.scatter(X_rm, y)
            plt.scatter(X_rm, price_by_k_and_b)
            plt.show()

    k_gradient = partial_k(X_rm, y, price_by_k_and_b)
    b_gradient = partial_b(X_rm, y, price_by_k_and_b)

    current_k = current_k + (-k_gradient) * learning_rate
    current_b = current_b + (-b_gradient) * learning_rate
    if i % 300 == 0:
        a = 0

print('2000次计算后的最佳值：get best_k: {} best_b: {}, and the loss is: {}'.format(best_k, best_b, min_loss))
# 非常奇怪的是，得出的预测值和实际数据值是垂直的。
# 找初始化的数据，把数据截距的数据多减50。多次实验发现：随机函数实际非常容易把后赋值的数据大于前赋值的数据

# 疑问：
# 1、如何显示需要计算多少次，而不是固定次数。如何权衡过拟合和欠拟合
# 2、不同的损失函数使用时，有何不同,数据的脉络

mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+380+310")  # 调整窗口在屏幕上弹出的位置
