import cv2
import matplotlib.pyplot as plt

# read image
img = cv2.imread("../Images/flower.jpg")

# convert image color from BRG to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# show images
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("BRG Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.show()