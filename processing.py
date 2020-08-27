import os
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

img = cv2.imread('./good_samples/generated_plot_1296.png', 0)

# pixel distributin analysis
# plt.hist(img.ravel(), 256, [0,250])
# plt.show()


# upscale images
scale_factor = 3.0
new_width = int(img.shape[1] * scale_factor)
new_height = int(img.shape[0] * scale_factor)
new_dim = (new_width, new_height)

resized_img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

# gradient smoothed
gradient_smoothed = cv2.Sobel(resized_img, cv2.CV_64F, 1, 0, ksize=5)


# edge detection
blur = cv2.GaussianBlur(resized_img, (5,5), 0)
edges = cv2.Canny(blur, 100, 200)

images = [resized_img,gradient_smoothed, edges]

for i in range(0,len(images)):
    plt.imshow(images[i], 'gray')
    plt.show()

