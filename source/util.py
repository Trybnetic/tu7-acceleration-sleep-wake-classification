import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


def look_up(position, reverse_label_dict):
    if ~np.isnan(position):
        return reverse_label_dict[int(position)]
    else:
        return ''

look_up = np.vectorize(look_up)

def is_lying(position_label):
    return position_label.startswith("Lying")

is_lying = np.vectorize(is_lying)


def is_sloughed(position_label):
    return position_label.startswith("Slouched")

is_sloughed = np.vectorize(is_sloughed)


def med_filt_on_sec(arr, hz=100, n_sec=900):
    return np.repeat(median_filter(arr[::hz], size=n_sec),hz)


def get_longest_lying_episode(arr):
    cutpoints = np.concatenate(([0], np.where(np.diff(arr))[0], [arr.shape[0]-1]))

    for idx in np.flip(np.diff(cutpoints).argsort()):
        start = cutpoints[idx] + 1
        end = cutpoints[idx+1] + 1
        
        if np.all(arr[start:end]):
            return start,end