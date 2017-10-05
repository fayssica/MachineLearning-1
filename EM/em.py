#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/05 23:10
# @Author  : ZangBo
# @Site    : zangbo.me
# @File    : em.py
# @Software: Sublime Text

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# 定义迭代次数
MAX = 10

# 输入实验结果，实验次数共计5次。
experiment = {"H":np.array([5,9,8,4,7]), "T":np.array([5,1,2,6,3])}

# 随机初始化两个硬币的正面朝上的概率为0-1之间的两位小数，应避免两者相同。
A_h = round(np.random.random(), 2)
B_h = round(np.random.random(), 2)
H = {"A":A_h, "B":B_h}

# 定义E步函数
def E_step(H):
    A_h = H["A"]
    B_h = H["B"]
    Z_A_temp = A_h**experiment["H"] * (1 - A_h)**experiment["T"]
    Z_B_temp = B_h**experiment["H"] * (1 - B_h)**experiment["T"]
    Z_A = Z_A_temp / (Z_A_temp + Z_B_temp)
    Z_B = Z_B_temp / (Z_A_temp + Z_B_temp)
    Z = {"A":Z_A, "B":Z_B}
    return Z

# 定义M步函数
def M_step(Z):
    Z_A = Z["A"]
    Z_B = Z["B"]
    A_h_sum = np.dot(experiment["H"], Z_A)
    A_t_sum = np.dot(experiment["T"], Z_A)
    A_h = A_h_sum / (A_h_sum + A_t_sum)
    B_h_sum = np.dot(experiment["H"], Z_B)
    B_t_sum = np.dot(experiment["T"], Z_B)
    B_h = B_h_sum / (B_h_sum + B_t_sum)
    H = {"A":A_h, "B":B_h}
    return H

def output(x):
	if x > 0.5:
		y = "A"
	if x <= 0.5:
		y = "B"
	return y

# 进行迭代
for i in range(MAX):
    Z = E_step(H)
    H = M_step(Z)

# 打印结果
print("迭代次数：%d"%MAX)
print("硬币A正面朝上的概率：%.2f\n硬币B正面朝上的概率：%.2f" %(H["A"], H["B"]))
result = list(map(output, Z["A"]))
print("投掷硬币顺序："+" ".join(result))