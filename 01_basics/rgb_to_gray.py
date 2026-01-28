import cv2
import matplotlib.pyplot as plt

# read image
img = cv2.imread("../Images/flower.jpg")

# BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# BGR to gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# show image
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_gray, cmap="gray")
plt.title("Gray Image")
plt.axis("off")

plt.show()