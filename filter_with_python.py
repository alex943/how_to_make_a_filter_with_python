# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
import skimage
from skimage import io


# - flatten介绍
# fp = [[3, 1, 2], [1, 3, 2], [2, 1, 3]]
# fp.flatten()
# array([3, 1, 2, 1, 3, 2, 2, 1, 3])

# - 插值介绍
# 线性插值法算法
# https://zh.wikipedia.org/zh-hans/%E7%BA%BF%E6%80%A7%E6%8F%92%E5%80%BC
# 公式:
# f(x, xp, fp) = k * (x - fpx.left) + fpy.left
# k = (fpy.right - fpy.left) / (fpx.right - fpx.left)
def channel_adjust(channel, values):
    # flatten
    orig_size = channel.shape
    flat_channel = channel.flatten()
    adjusted = np.interp(
        flat_channel,
        np.linspace(0, 1, len(values)),
        values)

    # put back into image form
    return adjusted.reshape(orig_size)


def split_channel(im):
    r = im[:, :, 0]
    g = im[:, :, 1]
    b = im[:, :, 2]
    return r, g, b


'''
通过减去原图像模糊的部分，得到它的锐化图像

:param a, 原图锐化的系数
:param b, 对原图进行模糊的系数
'''
def sharpen(rgb, a, b, sigma=10):
    blurred = filters.gaussian(rgb, sigma=sigma, multichannel=True)
    sharper = np.clip(rgb * a - blurred * b, 0, 1.0)
    return sharper


def merge_channel(r, g, b):
    return np.stack([r, g, b], axis=2)


def expose_light(im):
    pass


# 插值数值
red_px = [0, 0.05, 0.11, 0.20, 0.35, 0.48,
          0.62, 0.77, 0.85, 0.93, 1]

green_px = [0, 0.09, 0.17, 0.21, 0.26, 0.37,
            0.55, 0.77, 0.87, 0.86, 0.81]

blue_px = [0, 0.02, 0.07, 0.16, 0.33, 0.54,
           0.71, 0.86, 0.95, 0.98, 1]


# 我们将图像像素从0-255转为0-1之间的规格化数据，规格化之后，处理起来会很方便
im = skimage.img_as_float(io.imread("city.jpg"))


r, g, b = split_channel(im)

red_adjusted = channel_adjust(r, red_px)

green_adjusted = channel_adjust(g, green_px)

blue_adjusted = channel_adjust(b, blue_px)


rgb = merge_channel(red_adjusted, green_adjusted, blue_adjusted)

final = sharpen(rgb, 1.3, 0.3)


plt.imshow(final)

plt.show()
