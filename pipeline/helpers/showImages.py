import cv2
import matplotlib.pyplot as plt
import numpy as np

def showImages(show, images, titles=None, list_cmaps=None, grid=False, num_cols=3, figsize=(10, 10),
                    title_fontsize=30):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    # only show if show passed
    if show:
        # if not given lists, convert to lists
        if not type(images) is list:
            images = [images]
        if not type(titles) is list:
            titles = [titles]
        list_images = images
        list_titles = titles

        for index, img in enumerate(list_images):
            list_images[index] = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        assert isinstance(list_images, list)
        assert len(list_images) > 0
        assert isinstance(list_images[0], np.ndarray)

        if list_titles is not None:
            assert isinstance(list_titles, list)
            assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

        if list_cmaps is not None:
            assert isinstance(list_cmaps, list)
            assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

        num_images = len(list_images)
        num_cols = min(num_images, num_cols)
        num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

        # Create a grid of subplots.
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

        # Create list of axes for easy iteration.
        if isinstance(axes, np.ndarray):
            list_axes = list(axes.flat)
        else:
            list_axes = [axes]

        for i in range(num_images):
            img = list_images[i]
            title = list_titles[i] if list_titles is not None else 'Image %d' % (i)
            cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

            list_axes[i].imshow(img, cmap=cmap)
            list_axes[i].set_title(title, fontsize=title_fontsize)
            list_axes[i].grid(grid)

        for i in range(num_images, len(list_axes)):
            list_axes[i].set_visible(False)

        for i in range(0,len(list_axes)):
            list_axes[i].set_xticks([])
            list_axes[i].set_yticks([])

        fig.tight_layout()
        _ = plt.show()
        # author: "stackoverflowuser2010"

def img_is_color(img):
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False