# importing packages
import matplotlib.pyplot as plt
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
from skimage import io

reference = io.imread('E:/dragonfly-patterner/data/all_images/INATRANDOM-634882.jpg')
image = io.imread('E:/dragonfly-patterner/data/all_images/INAT-698706-1.jpg')
image = io.imread('E:/dragonfly-patterner/data/all_images/INAT-522813-1.jpg')
# loading data
#reference = data.moon()
#image = data.camera()

# matching histograms
matched = match_histograms(image, reference,
                           multichannel=True, )

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1,
                                    ncols=3,
                                    figsize=(8, 3),
                                    sharex=True,
                                    sharey=True)
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

# displaying images
ax1.imshow(image)
ax1.set_title('Source image')
ax2.imshow(reference)
ax2.set_title('Reference image')
ax3.imshow(matched)
ax3.set_title('Matched image')

plt.tight_layout()
plt.show()

# displaying histograms.
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

for i, img in enumerate((image, reference, matched)):
    for c, c_color in enumerate(('red', 'green', 'blue')):
        img_hist, bins = exposure.histogram(img[..., c],
                                            source_range='dtype')
        axes[c, i].plot(bins, img_hist / img_hist.max())
        img_cdf, bins = exposure.cumulative_distribution(img[..., c])
        axes[c, i].plot(bins, img_cdf)
        axes[c, 0].set_ylabel(c_color)

axes[0, 0].set_title('Source image')
axes[0, 1].set_title('Reference image')
axes[0, 2].set_title('Matched image')