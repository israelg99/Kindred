import numpy as np


class Numputil(object):
    @staticmethod
    def dT(arr):
        return arr[np.newaxis].T

    @staticmethod
    def lT(arr):
        return Numputil.dT(np.array(arr))
