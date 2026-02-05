import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# read image
img = cv2.imread('../Images/flower.jpg')

# convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert to Gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create contrast function
def adjust_contrast(image, alpha):
    return np.clip(alpha * (image.astype(np.float32) - 128) + 128, 0, 255).astype('uint8')

# original image
good = img_gray.copy()

# high contrast
high = adjust_contrast(img_gray, 2.5)

# low contrast
low = adjust_contrast(img_gray, 0.3)

# create histogram
def compute_hist(image):
    return cv2.calcHist([image], [0], None, [256], [0,256])

hist_good = compute_hist(good)
hist_high = compute_hist(high)
hist_low = compute_hist(low)

# statistics
def contrast_stats(image, name):
    std = np.std(image)
    dynamic_range = (image.max() - image.min()) / 255.0
    michelson = (np.percentile(image, 99) - np.percentile(image, 1)) / \
                (np.percentile(image, 99) + np.percentile(image, 1) + 1e-6)
    
    if std < 30:
        label = "Low Contrast"
    elif std < 60:
        label = "Medium (Good)"
    else:
        label = "High Contrast"

    print(f"{name:12} | Std Dev: {std:6.1f} | Range: {dynamic_range:5.3f} | Michelson: {michelson:5.3f} | → {label}")
    return std, dynamic_range, michelson, label

print("\nCONTRAST STATISTICS")
print("-" * 90)
contrast_stats(low, "Low Contrast")
contrast_stats(good, "Good Contrast")
contrast_stats(high, "High Contrast")
print("-" * 90)

# show Images and Histograms
plt.figure(figsize=(14,8))

titles = ['Low Contrast', 'Good Contrast (Original)', 'High Contrast']
images = [low, good, high]
hists = [hist_low, hist_good, hist_high]
colors = ['purple', 'green', 'orange']
stds = [np.std(low), np.std(good), np.std(high)]

for i in range(3):
    # Images
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[i], fontsize=13, fontweight='bold', color=colors[i])
    plt.axis('off')

    # Histograms
    plt.subplot(2, 3, i+4)
    plt.plot(hists[i], color=colors[i], linewidth=2)
    plt.fill_between(range(256), hists[i].flatten(), color=colors[i], alpha=0.3)

    # contrast zones (spread)
    p10 = np.percentile(images[i], 10)
    p90 = np.percentile(images[i], 90)
    plt.axvspan(p10, p90, color=colors[i], alpha=0.2, label='90% Intensity Range')

    # Std Dev lines
    mean_val = np.mean(images[i])
    plt.axvline(mean_val, color='black', linestyle='-', linewidth=1.2)
    plt.axvline(mean_val - stds[i], color='black', linestyle=':', linewidth=1, alpha=0.7)
    plt.axvline(mean_val + stds[i], color='black', linestyle=':', linewidth=1, alpha=0.7,
                label=f'±1σ = {stds[i]:.0f}')
    
    plt.title(f'Histogram – {titles[i]}', color=colors[i])
    plt.xlabel('Intensity (0–255)')
    plt.ylabel('Pixel Count')
    plt.xlim(0, 255)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

plt.suptitle('Histogram Analysis of Contrast Levels', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
