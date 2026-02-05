import cv2
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

# read image
img = cv2.imread('../Images/flower.jpg')

# convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert to Graysacle
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create exposure function
def adjust_exposure(image, factor):
    return np.clip(image * factor, 0, 255).astype('uint8')

# original is "Good" exposure
good = img_gray.copy()

# High exposure(overexposured)
high = adjust_exposure(img_gray, 1.6)

# Low exposure(underexposured)
low = adjust_exposure(img_gray, 0.5)

# create histograms
def compute_hist(image):
    return cv2.calcHist([image], [0], None, [256], [0, 256])

hist_good = compute_hist(good)
hist_high = compute_hist(high)
hist_low = compute_hist(low)

# statistics
def exposure_stats(image, name):
    mean = np.mean(image)
    shadows = np.sum(image < 50) / image.size * 100
    highlights = np.sum(image > 200) / image.size * 100
    status = "Overexposed" if highlights > 35 else "Underexposed" if shadows >35 else "Good"
    print(f"{name:12} | Mean: {mean:6.1f} | Shadows: {shadows:5.1f}% | Highlights: {highlights:5.1f}% | → {status}")
    return mean, shadows, highlights, status

print("\nEXPOSURE STATISTICS")
print("-" * 80)
exposure_stats(low,  "Low Exp.")
exposure_stats(good, "Good Exp.")
exposure_stats(high, "High Exp.")
print("-" * 80)

# plot histogarms
plt.figure(figsize=(16,8))

titles = ['Low Exposure (Underexposed)', 'Good Exposure (Balanced)', 'High Exposure (Overexposed)']
images = [low, good, high]
hists = [hist_low, hist_good, hist_high]
colors = ['blue', 'green', 'red']
means = [np.mean(low), np.mean(good), np.mean(high)]

for i  in range(3):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=13, fontweight='bold', color=colors[i])
    plt.axis('off')

    # Histogram subplot
    plt.subplot(2, 3, i+4)
    plt.plot(hists[i], color=colors[i], linewidth=2)
    plt.fill_between(range(256), hists[i].flatten(), color=colors[i], alpha=0.3)
    
    # Exposure zones
    plt.axvspan(0, 50, color='navy', alpha=0.3, label='Shadows')
    plt.axvspan(50, 200, color='lightgray', alpha=0.2, label='Midtones')
    plt.axvspan(200, 255, color='yellow', alpha=0.3, label='Highlights')
    
    # Mean line
    plt.axvline(means[i], color='black', linestyle='--', linewidth=1.5, 
                label=f'Mean = {means[i]:.0f}')
    
    plt.title(f'Histogram – {titles[i].split(" ")[0]} Exposure', color=colors[i])
    plt.xlabel('Intensity')
    plt.ylabel('Pixel Count')
    plt.xlim(0, 255)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

plt.suptitle('Histogram Analysis of Exposure Levels', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()