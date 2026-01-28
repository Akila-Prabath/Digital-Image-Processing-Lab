import cv2
import matplotlib.pyplot as plt

# read image
img = cv2.imread("../Images/flower.jpg")

# convert to rgb
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert to gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(img_gray, cmap="gray")
plt.title("Gray Image")
plt.axis("off")

plt.subplot(2,2,3)
plt.hist(img_rgb.ravel(), 256, [0,256])
plt.title("Grayscale Histogram for Original image")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.subplot(2,2,4)
plt.hist(img_gray.ravel(), 256, [0,256])
plt.title("Grayscale Histogram for Gray image")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.show()