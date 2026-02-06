import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image using grayscale mode
img = cv2.imread("../Images/flower.jpg", 0)

# Gaussian noise
mean = 0
sigma = 25
gaussian = np.random.normal(mean, sigma, img.shape)

noisy = img + gaussian
noisy = np.clip(noisy, 0, 255).astype('uint8')

# show images
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(noisy, cmap='gray')
plt.title("Gaussian Noise")
plt.axis("off")

plt.show()