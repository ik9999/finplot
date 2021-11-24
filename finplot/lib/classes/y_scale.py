import numpy as np

class YScale:
    def __init__(self, scaletype, scalef):
        self.scaletype = scaletype
        self.set_scale(scalef)

    def set_scale(self, scale):
        self.scalef = scale

    def xform(self, y):
        if self.scaletype == 'log':
            y = 10**y
        y = y * self.scalef
        return y

    def invxform(self, y, verify=False):
        y /= self.scalef
        if self.scaletype == 'log':
            if verify and y <= 0:
                return -1e6
            y = np.log10(y)
        return y

