import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# read image
img = cv2.imread("../Images/flower.jpg")

# convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert to Gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create brightness function
def adjust_brightness(image, delta):
    return np.clip(image.astype(np.int16) + delta, 0, 255).astype('uint8')

# original image
good = img_gray.copy()

# high brightness
high = adjust_brightness(img_gray, +100)

# low brightness
low = adjust_brightness(img_gray, -100)

# create histogram
def compute_hist(image):
    return cv2.calcHist([image], [0], None, [256], [0, 256])

hist_good = compute_hist(good)
hist_high = compute_hist(high)
hist_low = compute_hist(low)

# brightness statistics
def brightness_stats(image, name):
    mean = np.mean(image)
    median = np.median(image)
    shadows = np.sum(image < 50) / image.size * 100
    highlights = np.sum(image > 200) / image.size * 100
    
    if mean < 70:
        label = "Low Brightness (Dark)"
    elif mean < 180:
        label = "Good Brightness"
    else:
        label = "High Brightness (Over-bright)"
    
    print(f"{name:14} | Mean: {mean:6.1f} | Median: {median:6.1f} | Shadows: {shadows:5.1f}% | Highlights: {highlights:5.1f}% | → {label}")
    return mean, median, shadows, highlights, label

print("\nBRIGHTNESS STATISTICS")
print("-" * 100)
brightness_stats(low,  "Low Bright.")
brightness_stats(good, "Good Bright.")
brightness_stats(high, "High Bright.")
print("-" * 100)

# show images and histograms
plt.figure(figsize=(14, 8))

titles = ['Low Brightness (Dark)', 'Good Brightness (Balanced)', 'High Brightness (Over-bright)']
images = [low, good, high]
hists = [hist_low, hist_good, hist_high]
colors = ['darkblue', 'green', 'gold']
means = [np.mean(low), np.mean(good), np.mean(high)]

for i in range(3):
    # Image
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[i], fontsize=13, fontweight='bold', color=colors[i])
    plt.axis('off')

    # Histogram
    plt.subplot(2, 3, i+4)
    plt.plot(hists[i], color=colors[i], linewidth=2)
    plt.fill_between(range(256), hists[i].flatten(), color=colors[i], alpha=0.3)
    
    # Brightness zones
    plt.axvspan(0, 70, color='navy', alpha=0.3, label='Dark Zone')
    plt.axvspan(70, 180, color='lightgreen', alpha=0.2, label='Midtone Zone')
    plt.axvspan(180, 255, color='yellow', alpha=0.3, label='Bright Zone')
    
    # Mean & Median lines
    mean_val = means[i]
    median_val = np.median(images[i])
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.8, label=f'Mean = {mean_val:.0f}')
    plt.axvline(median_val, color='orange', linestyle=':', linewidth=1.5, label=f'Median = {median_val:.0f}')
    
    plt.title(f'Histogram – {titles[i].split(" ")[0]} Brightness', color=colors[i])
    plt.xlabel('Intensity (0–255)')
    plt.ylabel('Pixel Count')
    plt.xlim(0, 255)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

plt.suptitle('Histogram Analysis of Brightness Levels ', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()