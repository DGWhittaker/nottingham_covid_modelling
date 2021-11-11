import os

import numpy as np


def save_np_to_file(data, save_path):
    """ Utility for saving numpy array to file (mainly used for testing purposes)"""
    path = os.path.dirname(save_path)
    if path != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Make sure the path exists
        np.save(save_path, data)
