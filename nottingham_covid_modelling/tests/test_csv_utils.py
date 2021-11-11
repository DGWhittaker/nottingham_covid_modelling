import os
import random
import sys

from nottingham_covid_modelling.lib._csv_utils import read_csv, write_to_csv


def test_read_write(tmp_path):
    """ Check write_to_csv works as expected """
    save_file = os.path.join(tmp_path, "random data.csv")
    columns = ['half', 'giraffe', 'wacky', 'airport', 'breath', 'reaction', 'dangerous', 'suit', 'excited', 'longing']
    save_data = [{k: random.uniform(-sys.float_info.min, sys.float_info.max) for k in columns} for i in range(10)]

    write_to_csv(save_data, save_file, columns)
    reload_data = read_csv(save_file)
    reload_column = reload_data[0]
    
    assert reload_column == columns
    reload_data = [{k: float(v[reload_column.index(k)]) for k in reload_column} for v in reload_data[1:]]
    assert str(save_data) == str(reload_data)
