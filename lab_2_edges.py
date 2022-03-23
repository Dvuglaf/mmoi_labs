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


path = 'C://Users//aizee//Desktop//ssau//8//mmoi//lab1//imgs//14_LENA.TIF'
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
    return result_gradient
    #show()


def laplasian(image, threshold1, threshold2, threshold3):
    filter_kernel_x_first = numpy.array([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]])
    filter_kernel_x_second = numpy.array([[1, 0, 1],
                                         [0, -4, 0],
                                         [1, 0, 1]]).astype(numpy.float64)
    filter_kernel_x_third = numpy.array([[1, 1, 1],
                                         [1, -8, 1],
                                         [1, 1, 1]]).astype(numpy.float64)

    filter_kernel_x_second *= 1/2
    filter_kernel_x_third *= 1/3

    res_1 = scipy.signal.convolve2d(image, filter_kernel_x_first,
                                    mode='same', boundary='symm', fillvalue=0).astype(int)
    res_2 = scipy.signal.convolve2d(image, filter_kernel_x_second,
                                    mode='same', boundary='symm', fillvalue=0).astype(int)
    res_3 = scipy.signal.convolve2d(image, filter_kernel_x_third,
                                    mode='same', boundary='symm', fillvalue=0).astype(int)

    res_1 = numpy.abs(res_1)
    res_2 = numpy.abs(res_2)
    res_3 = numpy.abs(res_3)

    thresholding = numpy.vectorize(lambda elem, threshold: 255 if elem >= threshold else 0)
    result_gradient_1 = thresholding(res_1, threshold1)
    result_gradient_2 = thresholding(res_2, threshold2)
    result_gradient_3 = thresholding(res_3, threshold3)

    fig = plt.figure()
    subplot(fig, 2, 5, 1, image, 'Исходное изображение')
    subplot(fig, 2, 5, 2, res_1, 'Оценка лапласиана 1')
    subplot(fig, 2, 5, 3, res_2, 'Оценка лапласиана 2')
    subplot(fig, 2, 5, 4, res_3, 'Оценка лапласиана 3')
    subplot(fig, 2, 5, 5, result_gradient_1, 'Контуры "Лапласиан 1"')
    subplot(fig, 2, 5, 6, result_gradient_2, 'Контуры "Лапласиан 2"')
    subplot(fig, 2, 5, 7, result_gradient_3, 'Контуры "Лапласиан 3"')
    create_histogram_plot(fig, 2, 5, 8, res_1, 'Гистограмма оценки лапласиана 1')
    create_histogram_plot(fig, 2, 5, 9, res_2, 'Гистограмма оценки лапласиана 2')
    create_histogram_plot(fig, 2, 5, 10, res_3, 'Гистограмма оценки лапласиана 3')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.6,
                        hspace=0.25)

    return result_gradient_1, result_gradient_2, result_gradient_3
    #show()


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
    #show()
    return result_gradient


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
    print(numpy.max(res))

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
    #show()
    return result_gradient


first = simple_gradient(original_image, 25)
second = laplasian(original_image, 50, 30, 35)
third = prewitt(original_image, 10)
fourth = matching_laplasian(original_image, 20)

fig = plt.figure()
subplot(fig, 2, 3, 1, first, "Простой градиент")
subplot(fig, 2, 3, 2, second[0], "Лаплассиан 1")
subplot(fig, 2, 3, 3, second[1], "Лаплассиан 2")
subplot(fig, 2, 3, 4, second[2], "Лаплассиан 3")
subplot(fig, 2, 3, 5, third, "Прюитт")
subplot(fig, 2, 3, 6, fourth, "Согласованный лапласиан")
show()

