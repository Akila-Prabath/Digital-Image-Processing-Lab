import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image
img = cv2.imread("../Images/flower.jpg")

# convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Poisson noise
vals = len(np.unique(img))
vals = 2 ** np.ceil(np.log2(vals))

noisy = np.random.poisson(img * vals) / float(vals)
noisy = np.clip(noisy, 0, 255).astype('uint8')

# Show
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(noisy)
plt.title("Poisson Noise")
plt.axis("off")

plt.show()
