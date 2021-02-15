import os
import random
import sys

import numpy as np
from nottingham_covid_modelling.lib._save_to_file import save_np_to_file


def test_svae_reload(tmp_path):
    """ Check save_np_to_file works as expected """
    save_file = os.path.join(tmp_path, "random data.npy")
    save_data = np.array([random.uniform(-sys.float_info.min, sys.float_info.max) for i in range(10000)])
    save_np_to_file(save_data, save_file)

    load_back = np.load(save_file)
    assert np.allclose(save_data, load_back)
