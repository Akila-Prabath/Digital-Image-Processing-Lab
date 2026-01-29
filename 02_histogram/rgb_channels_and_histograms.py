import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image
img = cv2.imread("../Images/flower.jpg")

# convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# split channels
r_channel = img_rgb[:,:,0]
g_channel = img_rgb[:,:,1]
b_channel = img_rgb[:,:,2]

# show image size
print(f"Image loaded: {img_rgb.shape}(Height x Width x Channels)")

# create histogram for each channel
def compute_hist(channel):
    return cv2.calcHist([channel], [0], None, [256], [0, 256])

hist_r = compute_hist(r_channel)
hist_g = compute_hist(g_channel)
hist_b = compute_hist(b_channel)

# show channel statistics
def channel_stats(channel, name, color):
    mean = np.mean(channel)
    std = np.std(channel)
    peak = np.argmax(channel)
    print(f'{name:5} Channel | Mean: {mean:6.1f} | Std: {std:5.1f} | Peak Intensity: {peak:3}')
    return mean, std

print('\n' + '='*70)
print('RGB CHANNEL STATISTICS')
print('-'*70)
mean_r, std_r = channel_stats(r_channel, 'Red', 'red')
mean_g, std_g = channel_stats(g_channel, 'Green', 'green')
mean_b, std_b = channel_stats(b_channel, 'Blue', 'blue')
print('-'*70)

# show images and histograms
plt.figure(figsize=(14, 8))

channels = [
    (r_channel, hist_r, 'Red',   'red',    mean_r),
    (g_channel, hist_g, 'Green', 'green',  mean_g),
    (b_channel, hist_b, 'Blue',  'blue',   mean_b)
]

for i, (ch, hist, name, color, mean_val) in enumerate(channels):
    # --- Channel Image ---
    plt.subplot(2, 3, i+1)
    # Create masked image: show only this channel
    masked = np.zeros_like(img_rgb)
    masked[:, :, i] = ch
    plt.imshow(masked)
    plt.title(f'{name} Channel Only', fontsize=14, fontweight='bold', color=color)
    plt.axis('off')

    # --- Histogram ---
    plt.subplot(2, 3, i+4)
    plt.plot(hist, color=color, linewidth=2, label=f'{name} Histogram')
    plt.fill_between(range(256), hist.flatten(), color=color, alpha=0.3)
    
    # Mean line
    plt.axvline(mean_val, color='black', linestyle='--', linewidth=2, 
                label=f'Mean = {mean_val:.1f}')
    
    # Intensity zones
    plt.axvspan(0, 85, color='darkred', alpha=0.2, label='Low')
    plt.axvspan(85, 170, color='orange', alpha=0.1, label='Mid')
    plt.axvspan(170, 255, color='yellow', alpha=0.2, label='High')
    
    plt.title(f'{name} Channel Histogram', fontsize=14, fontweight='bold', color=color)
    plt.xlabel('Intensity (0–255)')
    plt.ylabel('Number of Pixels')
    plt.xlim(0, 255)
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.suptitle('RGB Channel Analysis', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()