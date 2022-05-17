import copy

import numpy
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imshow, show
from scipy.signal import convolve2d


def subplot(figure, n, m, pos, arr, title):
    sb = figure.add_subplot(n, m, pos)
    sb.set_title(title)
    imshow(arr, cmap='gray')


# Наложение на фон(background) count-штук объектов(obj) с яркостью 128
def image_with_objects(background, count, obj):
    copy_background = copy.copy(background)

    # Массив непересекающихся индексов для вставки объектов
    positions = np.array([[3, 3], [3, 13], [3, 23], [3, 33], [3, 43], [3, 53], [3, 60],
                          [13, 3], [13, 13], [13, 23], [13, 33], [13, 43], [13, 53], [13, 60],
                          [23, 3], [23, 13], [23, 23], [23, 33], [23, 43], [23, 53], [23, 60],
                          [33, 3], [33, 13], [33, 23], [33, 33], [33, 43], [33, 53], [33, 60],
                          [43, 3], [43, 13], [43, 23], [43, 33], [43, 43], [43, 53], [43, 60],
                          [53, 3], [53, 13], [53, 23], [53, 33], [53, 43], [53, 53], [53, 60],
                          [60, 3], [60, 13], [60, 23], [60, 33], [60, 43], [60, 53], [60, 60]], dtype=np.int32
                         )

    i = 0
    while i != count:
        idx = positions[np.random.randint(0, len(positions) - 1)]
        if copy_background[idx[0]][idx[1]] == 128:  # Если на этом месте уже есть объект, то генерируем другой
            i -= 1
            continue
        copy_background[idx[0]][idx[1]] = 128
        i += 1

    result = convolve2d(copy_background, obj, mode='same', boundary='symm', fillvalue=0)
    return result


# Добавление нулевых строк сверху и снизу, нулевых столбцов слева и справа.
def add_zeros(image):
    copy_image = copy.copy(image)
    copy_image = np.insert(copy_image, 0, 0, axis=1)  # строка сверху
    copy_image = np.insert(copy_image, 0, 0, axis=0)  # столбец слева

    copy_image = np.insert(copy_image, copy_image.shape[0], 0, axis=1)  # строка снизу
    copy_image = np.insert(copy_image, copy_image.shape[0], 0, axis=0)  # столбец справа
    return copy_image


# Корреляционный метод распознавания объектов(obj) на изображении(image)
def correlation_method(image, obj):
    image_size = image.shape[0]
    x = copy.copy(image)
    x = add_zeros(x)
    R = np.zeros((image_size, image_size), dtype=np.float32)

    # Нахождение энергии t(obj)
    t_energy = 0
    for k in range(0, obj.shape[0]):
        for l in range(0, obj.shape[1]):
            t_energy += (obj[k][l] ** 2)

    for n in range(1, image_size + 1):
        for m in range(1, image_size + 1):
            nominator = 0  # B = x(n + k, m + l)*t(k, l)        (k, l) from D
            denominator = 0  # x_energy = {x(n + k, m + l)}^2   (k, l) from D

            for k in range(-1, 2, 1):  # {-1, 0, 1}
                for l in range(-1, 2, 1):  # {-1, 0, 1}
                    nominator += (x[n + k][m + l] * obj[k + 1][l + 1])
                    denominator += (x[n + k][m + l] ** 2)

            R[n - 1][m - 1] = nominator / np.sqrt(denominator * t_energy)

    return R


def main():
    OBJECTS_COUNT = 8
    IMAGE_SIZE = 64
    THRESHOLD = 0.93
    # Объекты
    T = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=np.int32)
    REV_T = np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=np.int32)

    black_background = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)

    # Изображения с объектами на черном фоне
    objects_t = image_with_objects(black_background, OBJECTS_COUNT, T)  # буквы Т размером 3х3
    objects_rev_t = image_with_objects(black_background, OBJECTS_COUNT, REV_T)  # перевернутые буквы Т размером 3х3
    objects_t_and_rev_t = objects_t + objects_rev_t  # буквы Т и перевернутые буквы Т размером 3х3

    noise_background = np.random.normal(0, np.sqrt(np.var(objects_t)), (IMAGE_SIZE, IMAGE_SIZE)).astype(
        np.int32)  # фон - белый шум(0,sigma(obj))

    # Исходные изображения
    image_none = copy.copy(noise_background)  # Фон - белый шум
    image_with_t = objects_t + noise_background  # Буквы Т на фоне
    image_with_rev_t = objects_rev_t + noise_background  # Перевернутые буквы Т на фоне
    image_with_t_and_rev_t = objects_t_and_rev_t + noise_background  # Т и перевернутые Т на фоне

    # Корреляционные поля
    R_image_none_1 = correlation_method(image_none, T)  # кор. поля для изображения без объектов
    R_image_none_2 = correlation_method(image_none, REV_T)

    R_image_with_t_1 = correlation_method(image_with_t, T)  # кор. поля для изображений с Т
    R_image_with_t_2 = correlation_method(image_with_t, REV_T)

    R_image_with_rev_t_1 = correlation_method(image_with_rev_t, T)  # кор. поля для изображений с rev Т
    R_image_with_rev_t_2 = correlation_method(image_with_rev_t, REV_T)

    R_image_with_t_and_rev_t_1 = correlation_method(image_with_t_and_rev_t, T)  # кор. поля для изображений с Т и rev T
    R_image_with_t_and_rev_t_2 = correlation_method(image_with_t_and_rev_t, REV_T)

    thresholding = numpy.vectorize(lambda elem, threshold: 255 if elem >= threshold else 0)  # пороговая обработка

    # Изображения корреляционных полей после пороговой обработки
    g_none_1 = thresholding(R_image_none_1, THRESHOLD)
    g_none_2 = thresholding(R_image_none_2, THRESHOLD)

    g_with_t_1 = thresholding(R_image_with_t_1, THRESHOLD)
    g_with_t_2 = thresholding(R_image_with_t_2, THRESHOLD)

    g_with_rev_t_1 = thresholding(R_image_with_rev_t_1, THRESHOLD)
    g_with_rev_t_2 = thresholding(R_image_with_rev_t_2, THRESHOLD)

    g_with_t_and_rev_t_1 = thresholding(R_image_with_t_and_rev_t_1, THRESHOLD)
    g_with_t_and_rev_t_2 = thresholding(R_image_with_t_and_rev_t_2, THRESHOLD)

    # Отображение
    fig_1 = plt.figure()
    subplot(fig_1, 2, 3, 1, noise_background, "Фон")
    subplot(fig_1, 2, 3, 4, image_none, "Фон с добавленными объектами")
    subplot(fig_1, 2, 3, 2, R_image_none_1 * 255, "Корреляционное поле для Т")
    subplot(fig_1, 2, 3, 5, R_image_none_2 * 255, "Корреляционное поле для перевернутых Т")
    subplot(fig_1, 2, 3, 3, g_none_1, "Пороговая обработка кор. поля для Т")
    subplot(fig_1, 2, 3, 6, g_none_2, "Пороговая обработка кор. поля для перевернутых Т")

    fig_2 = plt.figure()
    subplot(fig_2, 2, 3, 1, noise_background, "Фон")
    subplot(fig_2, 2, 3, 4, image_with_t, "Фон с добавленными объектами")
    subplot(fig_2, 2, 3, 2, R_image_with_t_1 * 255, "Корреляционное поле для Т")
    subplot(fig_2, 2, 3, 5, R_image_with_t_2 * 255, "Корреляционное поле для перевернутых Т")
    subplot(fig_2, 2, 3, 3, g_with_t_1, "Пороговая обработка кор. поля для Т")
    subplot(fig_2, 2, 3, 6, g_with_t_2, "Пороговая обработка кор. поля для перевернутых Т")

    fig_3 = plt.figure()
    subplot(fig_3, 2, 3, 1, noise_background, "Фон")
    subplot(fig_3, 2, 3, 4, image_with_rev_t, "Фон с добавленными объектами")
    subplot(fig_3, 2, 3, 2, R_image_with_rev_t_1 * 255, "Корреляционное поле для Т")
    subplot(fig_3, 2, 3, 5, R_image_with_rev_t_2 * 255, "Корреляционное поле для перевернутых Т")
    subplot(fig_3, 2, 3, 3, g_with_rev_t_1, "Пороговая обработка кор. поля для Т")
    subplot(fig_3, 2, 3, 6, g_with_rev_t_2, "Пороговая обработка кор. поля для перевернутых Т")

    fig_4 = plt.figure()
    subplot(fig_4, 2, 3, 1, noise_background, "Фон")
    subplot(fig_4, 2, 3, 4, image_with_t_and_rev_t, "Фон с добавленными объектами")
    subplot(fig_4, 2, 3, 2, R_image_with_t_and_rev_t_1 * 255, "Корреляционное поле для Т")
    subplot(fig_4, 2, 3, 5, R_image_with_t_and_rev_t_2 * 255, "Корреляционное поле для перевернутых Т")
    subplot(fig_4, 2, 3, 3, g_with_t_and_rev_t_1, "Пороговая обработка кор. поля для Т")
    subplot(fig_4, 2, 3, 6, g_with_t_and_rev_t_2, "Пороговая обработка кор. поля для перевернутых Т")
    show()


main()
