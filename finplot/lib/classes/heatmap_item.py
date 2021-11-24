import numpy as np
from pyqtgraph import QtCore, QtGui

from .. import definitions as defs
from .. import functions
from .fin_plot_item import FinPlotItem

class HeatmapItem(FinPlotItem):
    def __init__(self, ax, datasrc, rect_size=0.9, filter_limit=0, colmap=defs.colmap_clash, whiteout=0.0, colcurve=lambda x:pow(x,4)):
        self.rect_size = rect_size
        self.filter_limit = filter_limit
        self.colmap = colmap
        self.whiteout = whiteout
        self.colcurve = colcurve
        self.col_data_end = len(datasrc.df.columns)
        super().__init__(ax, datasrc, lod=False)

    def generate_picture(self, boundingRect):
        prices = self.datasrc.df.columns[self.datasrc.col_data_offset:self.col_data_end]
        h0 = (prices[0] - prices[1]) * (1-self.rect_size)
        h1 = (prices[0] - prices[1]) * (1-(1-self.rect_size)*2)
        rect_size2 = 0.5 * self.rect_size
        df = self.datasrc.df.iloc[:, self.datasrc.col_data_offset:self.col_data_end]
        values = df.values
        # normalize
        values -= np.nanmin(values)
        values = values / (np.nanmax(values) / (1+self.whiteout)) # overshoot for coloring
        lim = self.filter_limit * (1+self.whiteout)
        p = self.painter
        for t,row in enumerate(values):
            for ci,price in enumerate(prices):
                v = row[ci]
                if v >= lim:
                    v = 1 - self.colcurve(1 - (v-lim)/(1-lim))
                    color = self.colmap.map(v, mode='qcolor')
                    p.fillRect(QtCore.QRectF(t-rect_size2, price+h0, self.rect_size, h1), color)
