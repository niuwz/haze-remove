import cv2 as cv
import numpy as np
import os
from haze_remove import *

if __name__ == '__main__':
    files = os.listdir('images')
    BETA = 0.7
    K = 8
    RATE = 0.001
    SIZE = 15
    for name in files:
        filename = 'images/' + name
        result = Main(filename,BETA,K,RATE,SIZE)
        new_name = './results/' + name.split('_')[0] + '_result.jpg'
        print(new_name)
        # 保存文件时需进行反归一化操作
        cv.imwrite(new_name, result*255)
    print('已在文件夹内生成全部去雾图像')