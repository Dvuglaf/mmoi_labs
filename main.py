import numpy
from skimage.io import imread
from matplotlib import pyplot as plt
from skimage.io import imshow, show
from skimage.exposure import histogram, equalize_hist


def create_histogram_plot(figure, i, j, pos, arr, title):
    sub = figure.add_subplot(i, j, pos)
    hist, bins = histogram(arr)
    sub.set_ylabel('Значение')
    sub.set_xlabel('Яркость')
    sub.set_title(title)
    plt.bar(bins, hist)
    plt.xlim([-1, 256])


def create_function_plot(figure, title, y, x_name, y_name, i=1, j=1, pos=1):
    sub = figure.add_subplot(i, j, pos)
    sub.set_title(title)
    sub.set_ylabel(y_name)
    sub.set_xlabel(x_name)
    sub.plot(y, color='blue')
    plt.xlim([0, 256])


def help_thresholding(elem, threshold):
    if elem >= threshold:
        return 255
    else:
        return 0


path = 'C://Users//aizee//Desktop//ssau//8//mmoi//lab1//imgs//14_LENA.TIF'
image = imread(path)

# @params:
#   data array
#   threshold value
thresholding = numpy.vectorize(help_thresholding)
image_thr = thresholding(image, 80)

fig = plt.figure(figsize=(10, 5))
sub = fig.add_subplot(2, 2, 1)
sub.set_title('Исходное изображение')
imshow(image, cmap='gray', vmin=0, vmax=255)

create_histogram_plot(fig, 2, 2, 3, image, 'Исходная гистограмма')

sub = fig.add_subplot(2, 2, 2)
sub.set_title('Изображение после пороговой обработки')
imshow(image_thr, cmap='gray', vmin=0, vmax=255)

create_histogram_plot(fig, 2, 2, 4, image_thr, 'Гистограмма после пороговой обработки')
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
show()

fig = plt.figure(figsize=(10, 5))

# x - аргумент функции g
x = numpy.arange(0, 255, 1)
create_function_plot(fig, 'График функции поэлементного преобразования (пороговая обработка)',
                     thresholding(x, 80), 'Яркость', 'Значение')
show()




# вычисление параметров линейного контрастирования (а, b)
def compute_for_linear_contrast(f_min, f_max):
    a = 255 / (f_max - f_min)
    b = -(255 * f_min) / (f_max - f_min)
    return [a, b]


params = compute_for_linear_contrast(image.min(), image.max())
print(image.min())
print(image.max())

# @params:
#   data array
#   param a
#   param b
linear_contrast = numpy.vectorize(lambda elem, a, b: a * elem + b)
image_lc = linear_contrast(image, params[0], params[1])
print(image_lc.min())
print(image_lc.max())

fig = plt.figure(figsize=(10, 5))
sub = fig.add_subplot(2, 2, 1)
sub.set_title('Исходное изображение')
imshow(image, cmap='gray', vmin=0, vmax=255)

create_histogram_plot(fig, 2, 2, 3, image, 'Исходная гистограмма')

sub = fig.add_subplot(2, 2, 2)
sub.set_title('Контрастированное изображение')
imshow(image_lc, cmap='gray', vmin=0, vmax=255)

create_histogram_plot(fig, 2, 2, 4, image_lc, 'Гистограмма после контрастирования')
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
show()

fig = plt.figure(figsize=(10, 5))
# x - аргумент функции g
x = numpy.arange(0, 255, 1)
create_function_plot(fig, 'График функции поэлементного преобразования (линейное контрастирование)',
                     linear_contrast(x, params[0], params[1]), 'Яркость', 'Значение')
show()



# возвращает значения функции плотности распределения
def compute_distribution(arr, count_of_pixels):
    cumsum_arr = numpy.cumsum(arr)
    div = numpy.vectorize(lambda x: x / count_of_pixels)
    return div(cumsum_arr)


hist, bins = numpy.histogram(image.flatten(), 256, [0, 256])
# Плотность распределения яркостей
distribution = compute_distribution(hist, image.shape[0] * image.shape[1])
# Поэлементное преобразование; плотность соответсвующего значения яркости умножается на 255
# @params:
#   data array
#   distribution
hist_equalization = numpy.vectorize(lambda brightness, distrib: round(distrib[brightness] * 255), excluded=[1])
image_he = hist_equalization(image, distribution)

fig = plt.figure(figsize=(10, 5))
sub = fig.add_subplot(2, 3, 1)
sub.set_title('Исходное изображение')
imshow(image, cmap='gray', vmin=0, vmax=255)

create_histogram_plot(fig, 2, 3, 4, image, 'Гистограмма исходного изображения')

sub = fig.add_subplot(2, 3, 2)
sub.set_title('Изображение после эквализации')
imshow(image_he, cmap='gray', vmin=0, vmax=255)

create_histogram_plot(fig, 2, 3, 5, image_he, 'Гистограмма изображения после эквализации')

sub = fig.add_subplot(2, 3, 3)
sub.set_title('Изображение после эквализации (skimage)')
eq = numpy.asarray(equalize_hist(image) * 255, dtype='uint8')
imshow(eq, cmap='gray', vmin=0, vmax=255)

create_histogram_plot(fig, 2, 3, 6, eq, 'Гистограмма изображения после эквализации (skimage)')

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.657,
                    hspace=0.4)

fig = plt.figure(figsize=(10, 5))
create_function_plot(fig, 'График интегральной функции распределения яркости (до)', distribution,
                     'Яркость', 'Значение функции распределения яркости', 2, 2, 1)

hist, bins = numpy.histogram(image_he.flatten(), 256, [0, 256])
distribution_output = compute_distribution(hist, image_he.shape[0] * image_he.shape[1])

create_function_plot(fig, 'График интегральной функции распределения яркости (после)', distribution_output,
                     'Яркость', 'Значение функции распределения яркости', 2, 2, 2)

hist, bins = numpy.histogram(image.flatten(), 256, [0, 256])
distribution = compute_distribution(hist, image.shape[0] * image.shape[1])
x = numpy.arange(0, 256, 1)
create_function_plot(fig, 'График функции поэлементного преобразования (эквализация)', hist_equalization(x, distribution),
                     'Яркость', 'Значение', 2, 2, 3)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.657,
                    hspace=0.4)
show()

# Массив гистограм исходного изображения
hist, bins = numpy.histogram(image.flatten(), 256, [0, 256])
print("Histogram:")
print(hist)
print("Bins:")
print(bins)











