from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch


def hide_axis(ax, with_stick=True):
    if with_stick:
        # stick visible but not the labels
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
    else:
        ax.set(yticklabels=[])
        ax.set(xticklabels=[])
    ax.tick_params(axis='both', which='both', length=0)

def plot_image(ax, image, title='', with_stick=True):
    ax.imshow(image)
    ax.set(title=title)
    hide_axis(ax, with_stick=with_stick)

def plot_raw_image(ax, image, title=''):
    ax.imshow(image, resample=False, interpolation='nearest')
    ax.set_title(title, fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])

def img(arr):
    if isinstance(arr, torch.Tensor):
        if len(arr.shape) == 4:
            arr = arr.squeeze(0)
        elif len(arr.shape) == 2:
            arr = arr.unsqueeze(0)

        arr = arr.permute(1, 2, 0).detach().cpu().numpy()

    assert isinstance(arr, np.ndarray)
    if len(arr.shape) == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr.clip(0, 1) * 255)

    return Image.fromarray(arr.astype(np.uint8)).convert('RGB')

def draw_border(img, color, width):
    a = np.asarray(img)
    for k in range(width):
        a[k, :] = color
        a[-k-1, :] = color
        a[:, k] = color
        a[:, -k-1] = color
    return Image.fromarray(a)