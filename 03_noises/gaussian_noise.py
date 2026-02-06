import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image using grayscale mode
img = cv2.imread("../Images/flower.jpg")

# convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Gaussian noise
mean = 0
sigma = 50
gaussian = np.random.normal(mean, sigma, img.shape)

noisy = img + gaussian
noisy = np.clip(noisy, 0, 255).astype('uint8')

# show images
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(noisy)
plt.title("Gaussian Noise")
plt.axis("off")

plt.show()