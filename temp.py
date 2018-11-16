from scipy import misc
import matplotlib.pyplot as plt
import cv2
import numpy as np

path_mask = './data/train_mask/1000002_l.bmp'
mask = misc.imread(path_mask)
print("misc data shape:", mask.shape)

img_cv = cv2.imread(path_mask)
print("cv data shape:", img_cv.shape)

kernel = np.ones((5, 5), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=3)

mask[mask < 128] = 1
mask[mask > 129] = 0

mask = 255*mask
plt.imshow(mask, cmap='gray', interpolation='bicubic')
plt.show()
print(mask)