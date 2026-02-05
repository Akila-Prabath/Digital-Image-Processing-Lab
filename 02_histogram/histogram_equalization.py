import cv2
import matplotlib.pyplot as plt

# read image
img = cv2.imread("../Images/flower.jpg", 0)

# creat histogram equalization
eq = cv2.equalizeHist(img)

# show plot
plt.figure(figsize=(12,6))

# original image
plt.subplot(2,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

# Equalized Image
plt.subplot(2,2,2)
plt.imshow(eq, cmap="gray")
plt.title("Histogram Equalized")
plt.axis("off")

# Histogram of Original Image
plt.subplot(2, 2, 3)
plt.hist(img.ravel(), 256, [0, 256], color='black')
plt.title("Original Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

# Histogram of Equalized Image
plt.subplot(2, 2, 4)
plt.hist(eq.ravel(), 256, [0, 256], color='black')
plt.title("Equalized Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
