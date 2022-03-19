import json
import numpy
from skimage import filters, color
from skimage.io import imread, imsave
from matplotlib import pyplot as plt
from skimage.io import imshow, show
from skimage.exposure import histogram

# JSON string
settings = {
    'path_to_image': 'C:/Users/aizee/Pictures/car.jpg',
    'algorithm': 'roberts',
    'path_to_result_image': 'C:/Users/aizee/Pictures/result.jpg'
}

# Save JSON string to vars.json and save setting into json_data
with open('lab_0_vars.json', 'w') as fp:
    json.dump(settings, fp)
with open('lab_0_vars.json') as json_file:
    json_data = json.load(json_file)

path = json_data['path_to_image']
algorithm = json_data['algorithm']

# Read image from path
image = imread(path)

# Set new value to path
path = json_data['path_to_result_image']

# Convert image to gray
gray_image = color.rgb2gray(image)

# Use selected algorithm, if algorithm is wrong, raise exception
if algorithm == 'prewitt':
    edges = filters.prewitt(gray_image)
elif algorithm == 'sobel':
    edges = filters.sobel(gray_image)
elif algorithm == 'roberts':
    edges = filters.roberts(gray_image)
elif algorithm == 'scharr':
    edges = filters.scharr(gray_image)
else:
    raise ValueError('Wrong')

# Display image and edges of image
fig = plt.figure(figsize=(10, 5))
fig.add_subplot(2, 2, 1)
imshow(image)
fig.add_subplot(2, 2, 2)
imshow(edges)

# Create subplot and display histogram of image
sb = fig.add_subplot(2, 2, 3)
hist_red, bins_red = histogram(image[..., 2])
hist_green, bins_green = histogram(image[..., 1])
hist_blue, bins_blue = histogram(image[..., 0])
sb.set_ylabel('число отсчетов')
sb.set_xlabel('значение яркости')
sb.set_title('Распределение яркостей по каждому каналу')
sb.plot(bins_green, hist_green, color='green', linestyle='-', linewidth=1)
sb.plot(bins_red, hist_red, color='red', linestyle='-', linewidth=1)
sb.plot(bins_blue, hist_blue, color='blue', linestyle='-', linewidth=1)
sb.legend(['green', 'red', 'blue'])

# Create subplot and display histogram of edges of image
sb = fig.add_subplot(2, 2, 4)
hist_red, bins_red = histogram(edges[..., 2])
hist_green, bins_green = histogram(edges[..., 1])
hist_blue, bins_blue = histogram(edges[..., 0])
sb.set_ylabel('число отсчетов')
sb.set_xlabel('значение яркости')
sb.set_title('Распределение яркости по каждому каналу')
sb.plot(bins_green, hist_green, color='green', linestyle='-', linewidth=1)
sb.plot(bins_red, hist_red, color='red', linestyle='-', linewidth=1)
sb.plot(bins_blue, hist_blue, color='blue', linestyle='-', linewidth=1)
sb.legend(['green', 'red', 'blue'])
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.8,
                    hspace=0.4)

# Save edges of image in file
imsave(path, (edges*255).astype(numpy.uint8))

# Display figure
show()
