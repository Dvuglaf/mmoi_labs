import numpy as np
from skimage.io import imread, imshow, show
import math
import matplotlib.pyplot as plt


IMAGE_SIZE = 256
IMAGE = imread('C://Users//aizee//OneDrive//Desktop//ssau//8//mmoi//lab1//imgs//14_LENA.TIF')


# Знак числа
def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    elif x < 0:
        return -1


# Вычисление энтропии изображения
def shannon_entropy(image):
    pixels_count = {}  # элемент: число вхождений элемента

    for row in image:
        for pixel in row:
            if pixels_count.get(pixel) is None:
                pixels_count[pixel] = 0
            pixels_count[pixel] += 1

    shannon_ent = 0.0
    for key in pixels_count:
        prob = float(pixels_count[key]) / (image.shape[0] * image.shape[1])  # вероятность появления
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent


# Кодер
# x - изображение
# e - макс. погрешность
# r - номер предсказателя
def diff_code(x, e, r):
    # Отсчеты ИХ предсказателя
    A = []
    if r == 1:
        A = [1, 0, 0, 0]
    elif r == 2:
        A = [0.5, 0.5, 0, 0]
    elif r == 3:
        A = [0.25, 0.25, 0.25, 0.25]
    elif r == 4:
        A = [1, 1, -1, 0]

    p = np.zeros((IMAGE_SIZE, IMAGE_SIZE))  # предсказание
    y = np.zeros((IMAGE_SIZE, IMAGE_SIZE))  # восстановление
    f = np.zeros((IMAGE_SIZE, IMAGE_SIZE))  # разностный сигнал (x - p)
    q = np.zeros((IMAGE_SIZE, IMAGE_SIZE))  # квантованное f

    for m in range(0, IMAGE_SIZE):
        for n in range(0, IMAGE_SIZE):
            # Подсчет предсказанного значения
            if n == 0:  # левый столбец
                p[m][n] = 0
            elif m == 0:  # верхняя строка
                p[m][n] = y[m][n-1]*A[0]
            elif n == IMAGE_SIZE - 1:  # нижняя строка
                p[m][n] = y[m][n-1]*A[0] + y[m-1][n]*A[1] + y[m-1][n-1]*A[2]
            else:
                p[m][n] = y[m][n-1]*A[0] + y[m-1][n]*A[1] + y[m-1][n-1]*A[2] + y[m-1][n+1]*A[3]

            f[m][n] = x[m][n] - p[m][n]  # вычисл. разностного сигнала
            q[m][n] = sign(f[m][n]) * ((abs(f[m][n]) + e)//(2*e + 1))  # квантование f
            y[m][n] = p[m][n] + q[m][n] * (2*e + 1)  # восстановление

    return q, f


# Декодер
# q - квантованные разности
# e - макс. погрешность
# r - номер предсказателя
def diff_decode(q, e, r):
    # Отсчеты ИХ предсказателя
    A = []
    if r == 1:
        A = [1, 0, 0, 0]
    elif r == 2:
        A = [0.5, 0.5, 0, 0]
    elif r == 3:
        A = [0.25, 0.25, 0.25, 0.25]
    elif r == 4:
        A = [1, 1, -1, 0]

    p = np.zeros((IMAGE_SIZE, IMAGE_SIZE))  # предсказание
    y = np.zeros((IMAGE_SIZE, IMAGE_SIZE))  # декомпрес. изображение
    for m in range(0, IMAGE_SIZE):
        for n in range(0, IMAGE_SIZE):
            if n == 0:
                p[m][n] = 0
            elif m == 0:
                p[m][n] = y[m][n-1]*A[0]
            elif n == IMAGE_SIZE - 1:
                p[m][n] = y[m][n-1]*A[0] + y[m-1][n]*A[1] + y[m-1][n-1]*A[2]
            else:
                p[m][n] = y[m][n-1]*A[0] + y[m-1][n]*A[1] + y[m-1][n-1]*A[2] + y[m-1][n+1]*A[3]

            y[m][n] = p[m][n] + q[m][n] * (2*e + 1)  # восстановление

    return y


# График зависимости энтропии массива q (это оценка объёма сжатых данных) от макс. погрешности e=0..50
def show_graphics():
    def get_values(r):
        e = np.arange(0, 51, 1)
        entropy_q = []
        for i in e:
            q, f = diff_code(IMAGE, i, r)
            entropy_q.append(shannon_entropy(q))
        return e, entropy_q

    e_1, entropy_q_1 = get_values(1)
    e_2, entropy_q_2 = get_values(2)

    fig = plt.figure()
    sub = fig.add_subplot(1, 1, 1)
    sub.set_title("График зависимости энтропии q от e")
    sub.set_ylabel("Энтропия q")
    sub.set_xlabel("e")
    sub.plot(e_1, entropy_q_1, color="blue", label='r=1')
    sub.plot(e_2, entropy_q_2, color="green", label='r=2')
    sub.legend()
    show()


# Примеры де компрессированных изображений для e=5, e=10, e=20, e=40
def show_decoded_examples(r):
    fig = plt.figure()
    sb = fig.add_subplot(2, 3, 1)
    sb.set_title("Оригинал")
    imshow(IMAGE, cmap='gray')

    pos = 2
    for e in [5, 10, 20, 40]:
        q, f = diff_code(IMAGE, e, r)
        y = diff_decode(q, e, r)

        sb = fig.add_subplot(2, 3, pos)
        sb.set_title("Восстановленное при e = " + str(e) + ", r = " + str(r))
        imshow(y, cmap='gray')

        pos += 1


# Разностный сигнал f (изображение) при e=0
def show_f(r):
    q, f = diff_code(IMAGE, 0, r)
    fig = plt.figure()
    sb = fig.add_subplot(1, 1, 1)
    sb.set_title("Разностное изображение f при e = 0, r = " + str(r))
    imshow(f, cmap='gray')


# Квантованный сигнал q (изображение) при e=0, e=5, e=10
def show_q(r):
    fig = plt.figure()
    pos = 1
    for e in [0, 5, 10]:
        q, f = diff_code(IMAGE, e, r)
        sb = fig.add_subplot(1, 3, pos)
        sb.set_title("Квантованный сигнал q при e =" + str(e) + ', r = ' + str(r))
        imshow(q, cmap='gray')
        pos += 1


# Контроль макс. ошибки
def check_max_error(e, r):
    q, f = diff_code(IMAGE, e, r)
    y = diff_decode(q, e, r)
    print("Максимальное значение x - y: ", np.max(IMAGE - y))
    print("e: ", e)
    if np.max(IMAGE - y) >= e:
        print("Контроль макс. ошибки выполняется для e = ", e)
    else:
        print("Контроль макс. ошибки не выполняется для e = ", e)


def main():
    # График зависимости энтропии массива q от максимальной погрешности e=0..50 при r=1 и r=2
    # show_graphics()

    # Примеры де компрессированных изображений для e=5, e=10, e=20, e=40
    show_decoded_examples(1)

    # Разностный сигнал f(изображение) при e=0
    show_f(1)

    # Квантованный сигнал q (изображение) при e=0, e=5, e=10
    show_q(1)
    show()


    # Контроль макс. ошибки
    check_max_error(49, 1)

main()