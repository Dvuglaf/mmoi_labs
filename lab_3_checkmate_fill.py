import numpy as np
from skimage.io import imsave, imshow, show, imread
from matplotlib import pyplot as plt


MAX_BRIGHTNESS_VALUE = 160
MIN_BRIGHTNESS_VALUE = 96
CELL_HEIGHT = 16
IMAGE_HEIGHT = 128
IMAGE_LENGTH = 128
BORDER_PROCESSING_PARAMETER = 1
VALUE_OF_ONE = 1


def border_processing_function(element_value):
    if element_value >= BORDER_PROCESSING_PARAMETER:
        return MAX_BRIGHTNESS_VALUE
    else:
        return MIN_BRIGHTNESS_VALUE


def border_processing(img_as_arrays):
    shape = np.shape(img_as_arrays)
    new_img_list = list(map(border_processing_function, np.reshape(img_as_arrays, img_as_arrays.size)))
    single_dimension_array = np.array(new_img_list)
    new_img = np.reshape(single_dimension_array, (shape[0], shape[1]))
    return new_img


def create_wb_histogram_plot(img_as_arrays):
    hist, bins = np.histogram(img_as_arrays.flatten(), 256, [0, 256])
    plt.plot(bins[:-1], hist, color='blue', linestyle='-', linewidth=1)


def create_chess_field_image():

    img = np.full((IMAGE_LENGTH, IMAGE_HEIGHT), 160).astype(np.uint8)
    line_index = 0
    indexes_of_start_black_odd = start_of_black_sells_odd()
    indexes_of_start_black_even = start_of_black_sells_even()
    odd_row = False

    while line_index < IMAGE_HEIGHT:

        column_index = 0

        if odd_row:
            current_indexes = indexes_of_start_black_odd
        else:
            current_indexes = indexes_of_start_black_even

        while column_index < IMAGE_LENGTH/(2*CELL_HEIGHT):
            j = 0
            while j < CELL_HEIGHT:
                img[line_index][current_indexes[column_index] + j] = 96
                j = j + 1
            column_index = column_index + 1
        line_index = line_index + 1
        if line_index % CELL_HEIGHT == 0:
            odd_row = not odd_row
    return img


def start_of_black_sells_odd():
    result = []
    i = 0
    j = 0
    while 2*i < IMAGE_HEIGHT:
        result.insert(j, 2*i)
        i = i + CELL_HEIGHT
        j = j + 1
    return result


def start_of_black_sells_even():
    result = []
    i = CELL_HEIGHT
    j = 0
    while i < IMAGE_HEIGHT:
        result.insert(j, i)
        i = i + 2*CELL_HEIGHT
        j = j + 1
    return result


chess_img = create_chess_field_image()
imsave("chess_board.jpg", chess_img)