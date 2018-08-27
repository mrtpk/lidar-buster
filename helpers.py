import matplotlib.pyplot as plt
import numpy as np

def display_stuff(imgs, labels, is_gray=False, figsize=(20,10), fontsize=30):
    '''
    Displays images and labels in a plot
    '''
    f, axes = plt.subplots(1, len(imgs), figsize=figsize)
    for i in range(0,len(imgs)):
        if is_gray:
            axes[i].imshow(imgs[i], cmap='gray')
        else:
            axes[i].imshow(imgs[i])
        axes[i].set_title(labels[i], fontsize=fontsize)
    return axes

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)
