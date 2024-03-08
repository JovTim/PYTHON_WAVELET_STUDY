import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data
import skimage as ski
from skimage.color import rgb2gray



#x, y = np.mgrid[0:1:128j, 0:1:128j]
#img = np.sin(2.0 * np.pi * 7 * x) + np.sin(2.0 * np.pi * 13 * y)

img = ski.data.astronaut()
img = rgb2gray(img)
coeffs2 = pywt.wavedec2(img, 'db1', level=1)
#coeffs2 = pywt.wavedec2(img, 'db1', level=1)

all_coeffs = np.concatenate([c.ravel() for sublist in coeffs2 for c in sublist if isinstance(c, np.ndarray)])

# Determine a threshold to retain a certain percentage of the strongest coefficients
percent = 10  # retain only the top 10% of coefficients
threshold = np.percentile(np.abs(all_coeffs), 100 - percent)

# Threshold the coefficients
coeffs2_thresholded = [(pywt.threshold(c, threshold, mode='soft') if isinstance(c, np.ndarray) else c) 
                       for c in coeffs2]

# Reconstruct the compressed image
compressed_img = pywt.waverec2(coeffs2_thresholded, 'db1')

# Plotting the original and compressed images
plt.figure(figsize=(5, 5))
plt.imshow(compressed_img, cmap='gray')
plt.axis('off')
plt.show()




#----------------EDGE------------
# Load image
#img_photo = pywt.data.camera()

#coeffs_photo = pywt.wavedec2(img_photo, 'db1', level=1)
#cA_photo, (cH_photo, cV_photo, cD_photo) = coeffs_photo

# Plotting the detail coefficients (edges)
#plt.figure(figsize=(5, 5))
#plt.imshow(np.abs(cV_photo), cmap='gray_r')
#plt.axis('off')
#plt.show()