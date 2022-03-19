import numpy
from matplotlib import pyplot as plt
from skimage.io import imread, imshow, show
import scipy.signal



path = 'C://Users//aizee//Desktop//ssau//8//mmoi//lab1//imgs//12_tank.tif'
image = imread(path)

filter_kernel_x = numpy.array([[-1, 1]])
filter_kernel_y =  numpy.array([[-1], [1]])

res_x = scipy.signal.convolve2d(image, filter_kernel_x,
                              mode='same', boundary='fill', fillvalue=0).astype(numpy.int)

res_y = scipy.signal.convolve2d(image, filter_kernel_y,
                              mode='same', boundary='fill', fillvalue=0).astype(numpy.int)


def help_e(s1, s2):
    return numpy.sqrt(s1 * s1 + s2 * s2)


e = numpy.vectorize(help_e)
res = e(res_x, res_y)


def help_thresholding(elem, threshold):
    if elem >= threshold:
        return 255
    else:
        return 0


thresholding = numpy.vectorize(help_thresholding)
image_thr = thresholding(res, 10)

fig = plt.figure()
imshow(image_thr)
show()