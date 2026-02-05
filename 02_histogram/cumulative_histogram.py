import cv2
import matplotlib.pyplot as plt

# read image
img = cv2.imread("../Images/flower.jpg")

# convert to gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# calculate histogram
hist = cv2.calcHist([img_gray], [0], None, [256], [0,256])

# calculate cumulative histogram
cum_hist = hist.cumsum()

# show plot
plt.figure(figsize=(15,6))

# Grayscale image
plt.subplot(1,3,1)
plt.imshow(img_gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

# Histogram of Grayscale Image
plt.subplot(1, 3, 2)
plt.hist(img_gray.ravel(), 256, [0, 256], color='black')
plt.title("Grayscale Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

# Cumulative Histogram
plt.subplot(1, 3, 3)
plt.plot(cum_hist, color='blue')
plt.title("Cumulative Histogram")
plt.xlabel("Intensity")
plt.ylabel("Cumulative Frequency")

plt.tight_layout()
plt.show()