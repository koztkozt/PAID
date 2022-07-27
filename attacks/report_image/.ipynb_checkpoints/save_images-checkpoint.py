from viewer_raids import draw_draw
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

# image number is 2112
# no_true =  0.0619
# adv_img = np.load("perturbed_image_FGSM.npy").transpose(1, 2, 3, 0)
# adv_img_pred = 0.2729
# adv_img_input = cv2.merge([adv_img[0],adv_img[0],adv_img[0]])
# adv_img_input = rescale_intensity(adv_img_input, in_range=(-1, 1), out_range=(0, 255))
# adv_img_image = draw_draw(adv_img_input, adv_img_pred, no_true)
# cv2.imwrite('FGSM.jpeg', adv_img_image)

# no_true =  0.0619
# adv_img = np.load("perturbed_image_Opt.npy").transpose(1, 2, 3, 0)
# adv_img_pred = 0.4035
# adv_img_input = cv2.merge([adv_img[0],adv_img[0],adv_img[0]])
# adv_img_input = rescale_intensity(adv_img_input, in_range=(-1, 1), out_range=(0, 255))
# adv_img_image = draw_draw(adv_img_input, adv_img_pred, no_true)
# cv2.imwrite('Opt.jpeg', adv_img_image)

# no_true =  0.0619
# adv_img = np.load("perturbed_image_OptU.npy").transpose(1, 2, 3, 0)
# adv_img_pred = 1.6959
# adv_img_input = cv2.merge([adv_img[0],adv_img[0],adv_img[0]])
# adv_img_input = rescale_intensity(adv_img_input, in_range=(-1, 1), out_range=(0, 255))
# adv_img_image = draw_draw(adv_img_input, adv_img_pred, no_true)
# cv2.imwrite('OptU.jpeg', adv_img_image)

# no_true =  0.0619
# adv_img = np.load("perturbed_image_advGAN.npy").transpose(1, 2, 3, 0)
# adv_img_pred = 0.3175
# adv_img_input = cv2.merge([adv_img[0],adv_img[0],adv_img[0]])
# adv_img_input = rescale_intensity(adv_img_input, in_range=(-1, 1), out_range=(0, 255))
# adv_img_image = draw_draw(adv_img_input, adv_img_pred, no_true)
# cv2.imwrite('advGAN.jpeg', adv_img_image)

# no_true =  0.0619
# adv_img = np.load("perturbed_image_advGANU.npy").transpose(1, 2, 3, 0)
# adv_img_pred = 0.4083
# adv_img_input = cv2.merge([adv_img[0],adv_img[0],adv_img[0]])
# adv_img_input = rescale_intensity(adv_img_input, in_range=(-1, 1), out_range=(0, 255))
# adv_img_image = draw_draw(adv_img_input, adv_img_pred, no_true)
# cv2.imwrite('advGANU.jpeg', adv_img_image)

adv_img = np.load("perturbed_image_none.npy").transpose(1, 2, 3, 0)
adv_img_pred = 0.0619
adv_img_input = cv2.merge([adv_img[0],adv_img[0],adv_img[0]])
adv_img_input = rescale_intensity(adv_img_input, in_range=(-1, 1), out_range=(0, 255))
adv_img_image = draw_draw(adv_img_input, adv_img_pred)
cv2.imwrite('None.jpeg', adv_img_image)