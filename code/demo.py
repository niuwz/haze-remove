import cv2 as cv
import numpy as np
import os
from haze_remove import *

if __name__ == '__main__':
    # 以天安门图像为例
    filename = '.\\images\\tiananmen_original.jpg'
    src = cv.imread(filename)
    # 定义部分参数
    BETA = 0.7
    K = 8
    RATE = 0.001
    SIZE = 15
    # 对图像进行归一化操作
    img = src.astype('float64')/255
    # 暗通道图像生成与结合
    min_dark = Min_Dark_Channel(img, SIZE)      # 最小值滤波
    mid_dark = Mid_Dark_Channel(img, SIZE)      # 中值滤波
    k_dark = Kmeans_Dark_Channel(img, SIZE, K)  # 聚类处理
    add_dark = Add(mid_dark, k_dark, BETA)      # 加权相加
    dark = Gauss_Add(add_dark, min_dark)        # 高斯加权
    A = Light_Channel(img, dark, RATE)
    te = Transmission_Estimate(img, A, SIZE)
    tx = Transmission_Refine(src, te)
    tx[tx < 0] = 0
    tx[tx > 1] = 1
    result = Recover(img, tx, A, 0.1)
    # 显示过程中及结果图像
    cv.imshow('original', src)
    cv.imshow('k-means dark', k_dark)
    cv.imshow('dark', dark)
    cv.imshow('tx', tx)
    cv.imshow('result', result)
    cv.waitKey(0)
