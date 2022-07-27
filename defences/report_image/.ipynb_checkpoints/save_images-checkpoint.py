import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity


img = np.load("perturbed_image_advGAN.npy").transpose(1, 2, 3, 0)
img_RGB = cv2.merge([img[0],img[0],img[0]])
img_RGB = rescale_intensity(img_RGB, in_range=(-1, 1), out_range=(0, 255))
img_RGB = cv2.resize(img_RGB, dsize=(320, 320),interpolation = cv2.INTER_AREA)
cv2.imwrite('perturbed_image_advGAN.jpeg', img_RGB)

img = np.load("squeeze_perturbed_image_bit_advGAN.npy").transpose(1, 2, 3, 0)
img_RGB = cv2.merge([img[0],img[0],img[0]])
img_RGB = rescale_intensity(img_RGB, in_range=(-1, 1), out_range=(0, 255))
img_RGB = cv2.resize(img_RGB, dsize=(320, 320),interpolation = cv2.INTER_AREA)
cv2.imwrite('squeeze_perturbed_image_bit_advGAN.jpeg', img_RGB)

img = np.load("squeeze_perturbed_image_blur_advGAN.npy").transpose(1, 2, 3, 0)
img_RGB = cv2.merge([img[0],img[0],img[0]])
img_RGB = rescale_intensity(img_RGB, in_range=(-1, 1), out_range=(0, 255))
img_RGB = cv2.resize(img_RGB, dsize=(320, 320),interpolation = cv2.INTER_AREA)
cv2.imwrite('squeeze_perturbed_image_blur_advGAN.jpeg', img_RGB)
