import copy

import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread, imshow, show
from skimage.util import random_noise
from scipy.signal import convolve2d
from scipy.ndimage import median_filter

from skimage.exposure import histogram


def subplot(figure, n, m, pos, arr, title):
    sb = figure.add_subplot(n, m, pos)
    sb.set_title(title)
    imshow(arr, cmap='gray', vmin=0, vmax=255)


def median(x, y):
    print("\tМедианная фильтрация")
    x_rate_m = median_filter(y, footprint=[[0, 1, 0], [1, 3, 1], [0, 1, 0]])
    mse_square = np.mean(np.square(x - x_rate_m))
    k_c = np.mean(np.square(x_rate_m - x)) / np.mean(np.square(y - x))
    print("\t\tКвадрат СКО фильтрации: ", mse_square)
    print("\t\tКоэффициент снижения шума: ", k_c)


def linear(x, y):
    print("\tЛинейный сглаживающий фильтр")
    kernel_1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.float64)
    kernel_2 = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]]).astype(np.float64)
    kernel_3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).astype(np.float64)

    kernel_1 *= 1 / 9
    kernel_2 *= 1 / 10
    kernel_3 *= 1 / 16

    x_rate_l = convolve2d(y, kernel_1, mode='same', boundary='symm', fillvalue=0)
    mse_square = np.mean(np.square(x - x_rate_l))
    k_c = np.mean(np.square(x_rate_l - x)) / np.mean(np.square(y - x))
    print("\t\tКвадрат СКО фильтрации: ", mse_square)
    print("\t\tКоэффициент снижения шума: ", k_c)


path = 'chess_board.jpg'
image = imread(path)
SNR = 10  # Сигнал/шум

Dx = np.var(image)  # Дисперсия изображения
Dv = Dx / SNR  # Дисперсия шума
print("Дисперсия исходного изображения: ", Dx)
print("Отношение сигнал/шум = 10:")
print("\tДисперсия аддитивного шума: ", Dv)
v = np.random.normal(0, np.sqrt(Dv), (128, 128)).astype(np.int8)
y = image + v

median(image, y)
linear(image, y)
SNR = 1
Dv = Dx / SNR  # Дисперсия шума
print("\nОтношение сигнал/шум = 1:")
print("\tДисперсия аддитивного шума: ", Dv)
v = np.random.normal(0, np.sqrt(Dv), (128, 128)).astype(np.int8)
y = image + v
median(image, y)
linear(image, y)


INTENSIVE = 0.1
print("\nИнтенсивность шума = 0.1:")
v = random_noise(image, 'salt', amount=INTENSIVE).astype(np.uint8) * 255
y = copy.deepcopy(image)
for i in range(0, 128):
    for j in range(0, 128):
        if v[i][j] == 255:
            y[i][j] = 255

median(image, y)
linear(image, y)

INTENSIVE = 0.3
print("\nИнтенсивность шума = 0.3:")
v = random_noise(image, 'salt', amount=INTENSIVE).astype(np.uint8) * 255
y = copy.deepcopy(image)
for i in range(0, 128):
    for j in range(0, 128):
        if v[i][j] == 255:
            y[i][j] = 255
median(image, y)
linear(image, y)





