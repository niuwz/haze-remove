import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 取最小值图像并进行开运算处理


def Min_Dark_Channel(img, size):
    # 对输入图像每个像素点取RGB最小值
    b, g, r = cv.split(img)
    dc = cv.min(cv.min(r, g), b)
    # 对最小值图像进行最小值滤波即腐蚀处理
    erode_kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    dark = cv.erode(dc, erode_kernel)
    # 腐蚀处理后进行膨胀处理
    dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2+size, 2+size))
    dark = cv.dilate(dark, dilate_kernel)
    return dark


# 对最小值图像进行中值滤波以获取对应暗通道图像
def Mid_Dark_Channel(img, size):
    # 获取RGB三通道最小值
    b, g, r = cv.split(img)
    dc = cv.min(cv.min(r, g), b)
    # 中值滤波要求图像格式为UINT8，故进行反归一化处理
    dc *= 255
    dc = dc.astype('uint8')
    # 中值滤波
    dark = cv.medianBlur(dc, size)
    # 归一化处理，以保持全局图像格式一致
    dark = dark.astype('float64')/255
    return dark


# K means处理图像并进行最小值滤波
def Kmeans_Dark_Channel(img, size, k=8):
    Z = img.reshape((-1, 3))
    # 函数要求图像格式为float32
    Z = Z.astype('float32')
    # 定义终止标准 聚类数并应用k均值
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k
    ret, label, center = cv.kmeans(
        Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center*255)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    res2 = res2.astype('float64')/255
    # 获取最小值图像并进行最小值滤波
    b, g, r = cv.split(res2)
    dc = cv.min(cv.min(r, g), b)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    kdark = cv.erode(dc, kernel)
    return kdark


def Add(dark, kdark, beta):
    return cv.add(dark * beta, kdark * (1 - beta))


# 高斯加权处理
def Gauss_Add(Fdark, mindark):
    a1 = 4
    a2 = 2
    z = 0.6

    gx = a2 * np.exp(-1*(1.5-Fdark)**2/z)
    Tx = gx + a1
    Ig = (a1*Fdark+gx*mindark)/Tx
    Ig = Ig.astype('float64')
    return Ig


def Light_Channel(sec_img, dark, rate):
    # 获取图像尺寸
    [h, w] = sec_img.shape[:2]
    size_of_img = h*w
    numpx = int(max(size_of_img * rate, 1))
    dark_vec = dark.reshape(1, -1)
    # 获取亮度阈值
    threshold = dark_vec[0, dark_vec.argsort()[0][-1 * numpx]]
    # 寻找符合条件的像素点的位置
    atmo = {}
    for x in range(h):
        for y in range(w):
            if dark[x, y] >= threshold:
                atmo.update({(x, y): np.mean(sec_img[x, y, :])})
    pos = sorted(atmo.items(), key=lambda item: item[1], reverse=True)[0][0]
    A = np.array([sec_img[pos[0], pos[1], :]])
    return A

# 估计透射率取值


def Transmission_Estimate(img, A, size):
    omega = 0.95
    img_empty = np.empty(img.shape, img.dtype)

    for ind in range(0, 3):
        img_empty[:, :, ind] = img[:, :, ind]/A[0, ind]

    transmission = 1 - omega*Min_Dark_Channel(img_empty, size)
    return transmission

# 使用导向滤波优化透射率


def Transmission_Refine(img, te):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guided_Filter(gray, te, r, eps)

    return t

# 导向滤波


def Guided_Filter(img, p, r, eps=1e-4):
    mean_I = cv.boxFilter(img, cv.CV_64F, (r, r))
    mean_p = cv.boxFilter(p, cv.CV_64F, (r, r))
    mean_Ip = cv.boxFilter(img*p, cv.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv.boxFilter(img*img, cv.CV_64F, (r, r))
    var_I = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv.boxFilter(a, cv.CV_64F, (r, r))
    mean_b = cv.boxFilter(b, cv.CV_64F, (r, r))

    q = mean_a*img + mean_b
    return q


def Recover(im, t, A, tx):
    res = np.empty(im.shape, im.dtype)
    t = cv.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind]-A[0, ind])/t + A[0, ind]

    return res


def Main(filename, BETA, K, RATE, SIZE):
    src = cv.imread(filename)
    img = src.astype('float64')/255
    min_dark = Min_Dark_Channel(img, SIZE)
    mid_dark = Mid_Dark_Channel(img, SIZE)
    k_dark = Kmeans_Dark_Channel(img, SIZE, K)
    add_dark = Add(mid_dark, k_dark, BETA)
    dark = Gauss_Add(add_dark, min_dark)
    A = Light_Channel(img, dark, RATE)
    te = Transmission_Estimate(img, A, SIZE)
    tx = Transmission_Refine(src, te)
    tx[tx < 0] = 0
    tx[tx > 1] = 1
    result = Recover(img, tx, A, 0.1)

    return result
