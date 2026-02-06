import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image 
img = cv2.imread("../Images/flower.jpg")

# convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

noisy = img.copy()

# probability
prob = 0.1

# Salt
num_salt = int(prob * img.size * 0.5)
coords = [np.random.randint(0, i, num_salt) for i in img.shape[:2]]
noisy[coords[0], coords[1], :] = 255     

# Pepper
num_pepper = int(prob * img.size * 0.5)
coords = [np.random.randint(0, i, num_pepper) for i in img.shape[:2]]
noisy[coords[0], coords[1], :] = 0

# Show images
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(noisy)
plt.title("Salt & Pepper Noise")
plt.axis("off")

plt.show()