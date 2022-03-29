import copy

import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread, imshow, show
from skimage.util import random_noise
from scipy.signal import convolve2d
from scipy.ndimage import median_filter


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
    return x_rate_m


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
    return x_rate_l


path = 'chess_board.jpg'
image = imread(path)
SNR = 10  # Сигнал/шум

Dx = np.var(image)  # Дисперсия изображения
Dv = Dx / SNR  # Дисперсия шума
print("Дисперсия исходного изображения: ", Dx)
print("Отношение сигнал/шум = 10:")
print("\tДисперсия аддитивного шума: ", Dv)
v_add_10 = np.random.normal(0, np.sqrt(Dv), (128, 128)).astype(int)
y_add_10 = image + v_add_10

x_rate_median_add_10 = median(image, y_add_10)
x_rate_linear_add_10 = linear(image, y_add_10)
SNR = 1
Dv = Dx / SNR  # Дисперсия шума
print("\nОтношение сигнал/шум = 1:")
print("\tДисперсия аддитивного шума: ", Dv)
v_add_1 = np.random.normal(0, np.sqrt(Dv), (128, 128)).astype(int)
y_add_1 = image + v_add_1
x_rate_median_add_1 = median(image, y_add_1)
x_rate_linear_add_1 = linear(image, y_add_1)


INTENSIVE = 0.1
print("\nИнтенсивность шума = 0.1:")
v_imp_0_1 = random_noise(np.full((128, 128), -1), 's&p', amount=INTENSIVE)
y_imp_0_1 = copy.deepcopy(image).astype(np.uint16)

# Добавление шума и изменения значений пикселов. Шум был {-1, 0, 1}, станет {0, 128, 255}
for i in range(0, 128):
    for j in range(0, 128):
        if v_imp_0_1[i][j] == 1:
            y_imp_0_1[i][j] = 255
            v_imp_0_1[i][j] = 255
        elif v_imp_0_1[i][j] == -1:
            y_imp_0_1[i][j] = 0
            v_imp_0_1[i][j] = 0
        else:
            v_imp_0_1[i][j] = 128

x_rate_median_imp_0_1 = median(image, y_imp_0_1)
x_rate_linear_imp_0_1 = linear(image, y_imp_0_1)

INTENSIVE = 0.3
print("\nИнтенсивность шума = 0.3:")
v_imp_0_3 = random_noise(np.full((128, 128), -1), 's&p', amount=INTENSIVE)
y_imp_0_3 = copy.deepcopy(image).astype(np.uint16)

# Добавление шума и изменения значений пикселов. Шум был {-1, 0, 1}, станет {0, 128, 255}
for i in range(0, 128):
    for j in range(0, 128):
        if v_imp_0_3[i][j] == 1:
            y_imp_0_3[i][j] = 255
            v_imp_0_3[i][j] = 255
        elif v_imp_0_3[i][j] == -1:
            y_imp_0_3[i][j] = 0
            v_imp_0_3[i][j] = 0
        else:
            v_imp_0_3[i][j] = 128
x_rate_median_imp_0_3 = median(image, y_imp_0_3)
x_rate_linear_imp_0_3 = linear(image, y_imp_0_3)

fig = plt.figure()
subplot(fig, 3, 3, 1, image, "Исходное изображение")
subplot(fig, 3, 3, 2, v_add_10 + 128, "Аддитивный шум, snr = 10")
subplot(fig, 3, 3, 3, v_add_1 + 128, "Аддитивный шум, snr = 1")
subplot(fig, 3, 3, 4, x_rate_median_add_10, "Медианная фильтрация, snr = 10")
subplot(fig, 3, 3, 5, y_add_10, "Зашумленное изображение, snr = 10")
subplot(fig, 3, 3, 6, y_add_1, "Зашумленное изображение, snr = 1")
subplot(fig, 3, 3, 7, x_rate_median_add_1, "Медианная фильтрация, snr = 1")
subplot(fig, 3, 3, 8, x_rate_linear_add_10, "Линейная фильтрация, snr = 10")
subplot(fig, 3, 3, 9, x_rate_linear_add_1, "Линейная фильтрация, snr = 1")
show()

fig = plt.figure()
subplot(fig, 3, 3, 1, image, "Исходное изображение")
subplot(fig, 3, 3, 2, v_imp_0_1, "Импульсный шум, p = 0.1")
subplot(fig, 3, 3, 3, v_imp_0_3, "Импульсный шум, p = 0.3")
subplot(fig, 3, 3, 4, x_rate_median_imp_0_1, "Медианная фильтрация, p = 0.1")
subplot(fig, 3, 3, 5, y_imp_0_1, "Зашумленное изображение, p = 0.1")
subplot(fig, 3, 3, 6, y_imp_0_3, "Зашумленное изображение, p = 0.3")
subplot(fig, 3, 3, 7, x_rate_median_imp_0_3, "Медианная фильтрация, p = 0.3")
subplot(fig, 3, 3, 8, x_rate_linear_imp_0_1, "Линейная фильтрация, p = 0.1")
subplot(fig, 3, 3, 9, x_rate_linear_imp_0_3, "Линейная фильтрация, p = 0.3")
show()
