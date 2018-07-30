# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np


def visual_interpolation_values_in_line_plot(fp):
    fx = np.linspace(0, 1, len(fp))
    plt.plot(fx, fp, '-r')
    plt.show()


def visual_img_in_hist(channel, tag='r'):
    plt.hist(channel.flatten(), bins=256, color=tag, align='right')
    plt.show()


def visual_img_in_number(channel):
    # im2 = im[:, :, 0] * 0.5 + im[:, :, 1] * 0.2 + im[:, :, 2] * 0.3
    np.savetxt("img_in_number.txt", channel, fmt='%3d', delimiter=' ')


def values_bulider():
    s = "["
    for i in range(128):
        s += str(0) + ","
    np.linspace(128, 255, 127)



'''
if __name__ == "__main__":
    values_bulider()
'''