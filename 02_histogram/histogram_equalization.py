import cv2
import matplotlib.pyplot as plt

# read image
img = cv2.imread("../Images/flower.jpg", 0)

# creat histogram equalization
eq = cv2.equalizeHist(img)

# show plot
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(eq, cmap="gray")
plt.title("Histogram Equalized")
plt.axis("off")

plt.show()
