import cv2
import matplotlib.pyplot as plt

# read image
img = cv2.imread("../Images/flower.jpg")

# convert image color from BRG to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# show image
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()