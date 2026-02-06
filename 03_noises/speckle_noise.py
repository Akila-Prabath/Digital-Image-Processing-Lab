import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image
img = cv2.imread("../Images/flower.jpg")

# convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Speckle noise (multiplicative)
gauss = np.random.randn(*img.shape)

noisy = img + img * gauss * 0.5
noisy = np.clip(noisy, 0, 255).astype('uint8')

# Show
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(noisy, cmap='gray')
plt.title("Speckle Noise")
plt.axis("off")

plt.show()
