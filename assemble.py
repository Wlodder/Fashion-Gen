import os
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

# pixel distributin analysis
# plt.hist(img.ravel(), 256, [0,250])
# plt.show()

# generate random RGB colour
def random_colour():
    R = np.random.randint(0, 256)
    G = np.random.randint(0, 256)
    B = np.random.randint(0, 256)
    return (R, G, B)


# set of individual samples 
generated_sample_set = [1026,1188, 756]

for sample in generated_sample_set:
    for i in range(0,10):
        # threshold and read each image
        img = cv2.imread("individual_samples/generated_sample_{sample_set:d}_{sample_no:d}.png".format(sample_set=sample, sample_no=i),0)        
        mask = cv2.cvtColor(np.uint8(img),cv2.COLOR_GRAY2RGB)

        mask = cv2.GaussianBlur(mask, (7,7), 0)
        edges = cv2.Canny(mask, 120, 200)
        edges = cv2.GaussianBlur(edges, (5,5), 0)
        

        # Pattern processing
        pattern = cv2.imread("patterns/Pattern_2.png")

        colour_1 = random_colour()
        colour_2 = random_colour()
        
        for j in range(pattern.shape[0]):
            for k in range(pattern.shape[1]):
                # if black 
                if pattern[j][k][0] == 255:
                    pattern[j][k] = colour_1
                else:
                    pattern[j][k] = colour_2
        
        pattern = cv2.GaussianBlur(pattern, (13,13), 3)
        
        # combining
        for j in range(pattern.shape[0]):
            for k in range(pattern.shape[1]):
                if edges[j][k]== 255:
                    pattern[j][k][0] = 255
                    pattern[j][k][1] = 255
                    pattern[j][k][2] = 255
                else:
                    pattern[j][k][0] = int(np.abs(mask[j][k][0] / 255 - 1) * pattern[j][k][0])
                    pattern[j][k][1] = int(np.abs(mask[j][k][1] / 255 - 1) * pattern[j][k][1])
                    pattern[j][k][2] = int(np.abs(mask[j][k][2] / 255 - 1) * pattern[j][k][2])
                
            
        plt.imshow(pattern)
        plt.show()

