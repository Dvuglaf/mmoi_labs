import numpy
from matplotlib import pyplot as plt
from skimage.io import imread, imshow, show
import scipy.signal
from skimage.exposure import histogram


def subplot(figure, n, m, pos, arr, title):
    sb = figure.add_subplot(n, m, pos)
    sb.set_title(title)
    imshow(arr, cmap='gray', vmin=0, vmax=255)


def create_histogram_plot(figure, n, m, pos, arr, title):
    sb = figure.add_subplot(n, m, pos)
    hist, bins = histogram(arr)
    sb.set_ylabel('Значение')
    sb.set_xlabel('Яркость')
    sb.set_title(title)
    plt.bar(bins, hist)
    plt.xlim([-1, 256])


path = 'C://Users//aizee//Desktop//ssau//8//mmoi//lab1//imgs//13_zelda.tif'
original_image = imread(path)


def simple_gradient(image, threshold):

    filter_kernel_x = numpy.array([[-1, 1]])
    filter_kernel_y = numpy.array([[-1], [1]])

    res_x = scipy.signal.convolve2d(image, filter_kernel_x,
                                  mode='same', boundary='symm', fillvalue=0).astype(int)
    res_y = scipy.signal.convolve2d(image, filter_kernel_y,
                                  mode='same', boundary='symm', fillvalue=0).astype(int)

    res_x = numpy.abs(res_x)
    res_y = numpy.abs(res_y)

    e = numpy.vectorize(lambda s1, s2: numpy.sqrt(s1 ** 2 + s2 ** 2))
    res = e(res_x, res_y).astype(int)

    thresholding = numpy.vectorize(lambda elem, threshold: 255 if elem >= threshold else 0)
    result_gradient = thresholding(res, threshold)


    fig = plt.figure()
    subplot(fig, 2, 3, 1, image, 'Исходное изображение')
    subplot(fig, 2, 3, 2, res_x, 'ЧП по горизонтали')
    subplot(fig, 2, 3, 3, res, 'Оценка градиента')
    subplot(fig, 2, 3, 4, result_gradient, 'Контуры "Простой градиент"')
    subplot(fig, 2, 3, 5, res_y, 'ЧП по вертикали')
    create_histogram_plot(fig, 2, 3, 6, res, 'Гистограмма оценки градиента')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.6,
                        hspace=0.25)
    show()


def laplasian(image, threshold):
    filter_kernel_x = numpy.array([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]])

    res = scipy.signal.convolve2d(image, filter_kernel_x,
                                    mode='same', boundary='symm', fillvalue=0).astype(int)
    res = numpy.abs(res)
    thresholding = numpy.vectorize(lambda elem, threshold: 255 if elem >= threshold else 0)
    result_gradient = thresholding(res, threshold)

    fig = plt.figure()
    subplot(fig, 2, 2, 1, image, 'Исходное изображение')
    subplot(fig, 2, 2, 2, res, 'Оценка лапласиана')
    subplot(fig, 2, 2, 3, result_gradient, 'Контуры "Лапласиан"')
    create_histogram_plot(fig, 2, 2, 4, res, 'Гистограмма оценки лапласиана')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.6,
                        hspace=0.25)
    show()


def prewitt(image, threshold):
    filter_kernel_x = numpy.array([[-1, -1, -1],
                                   [0, 0, 0],
                                   [1, 1, 1]]).astype(numpy.float64)
    filter_kernel_y = numpy.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]]).astype(numpy.float64)
    filter_kernel_x *= 1/6
    filter_kernel_y *= 1/6
    res_x = scipy.signal.convolve2d(image, filter_kernel_x,
                                    mode='same', boundary='symm', fillvalue=0).astype(int)
    res_y = scipy.signal.convolve2d(image, filter_kernel_y,
                                    mode='same', boundary='symm', fillvalue=0).astype(int)

    res_x = numpy.abs(res_x)
    res_y = numpy.abs(res_y)

    e = numpy.vectorize(lambda s1, s2: numpy.sqrt(s1 ** 2 + s2 ** 2))
    res = e(res_x, res_y).astype(int)

    thresholding = numpy.vectorize(lambda elem, threshold: 255 if elem >= threshold else 0)
    result_gradient = thresholding(res, threshold)

    fig = plt.figure()
    subplot(fig, 2, 3, 1, image, 'Исходное изображение')
    subplot(fig, 2, 3, 2, res_x, 's1')
    subplot(fig, 2, 3, 3, res, 'Оценка градиента')
    subplot(fig, 2, 3, 4, result_gradient, 'Контуры "Оператор Прюитт"')
    subplot(fig, 2, 3, 5, res_y, 's2')
    create_histogram_plot(fig, 2, 3, 6, res, 'Гистограмма оценки градиента')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.6,
                        hspace=0.25)
    show()


def matching_laplasian(image, threshold):
    filter_kernel = numpy.array([[2, -1, 2],
                                   [-1, -4, -1],
                                   [2, -1, 2]]).astype(numpy.float64)
    filter_kernel *= 1/3
    res = scipy.signal.convolve2d(image, filter_kernel,
                                  mode='same', boundary='symm', fillvalue=0).astype(int)
    res = numpy.abs(res)
    thresholding = numpy.vectorize(lambda elem, threshold: 255 if elem >= threshold else 0)
    result_gradient = thresholding(res, threshold)

    fig = plt.figure()
    subplot(fig, 2, 2, 1, image, 'Исходное изображение')
    subplot(fig, 2, 2, 2, res, 'Оценка лапласиана')
    subplot(fig, 2, 2, 3, result_gradient, 'Контуры "Согласование Лапласиан"')
    create_histogram_plot(fig, 2, 2, 4, res, 'Гистограмма оценки лапласиана')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.6,
                        hspace=0.25)
    show()


simple_gradient(original_image, 15)
laplasian(original_image, 30)
prewitt(original_image, 10)
matching_laplasian(original_image, 20)



