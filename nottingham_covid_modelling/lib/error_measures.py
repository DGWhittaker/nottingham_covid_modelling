import numpy as np


def calculate_RMSE(x, y):

    '''Returns root mean square error, RMSE'''

    return np.sqrt(calculate_MSE(x, y))


def calculate_MSE(x, y):

    '''Returns mean square error, MSE'''

    return np.square(np.subtract(x, y)).mean()
