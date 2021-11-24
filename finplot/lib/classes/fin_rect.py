import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from .. import definitions as defs
from .. import functions

class FinRect(pg.RectROI):
    def __init__(self, ax, brush, *args, **kwargs):
        self.ax = ax
        self.brush = brush
        super().__init__(*args, **kwargs)

    def paint(self, p, *args):
        r = QtCore.QRectF(0, 0, self.state['size'][0], self.state['size'][1]).normalized()
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.currentPen)
        p.setBrush(self.brush)
        p.translate(r.left(), r.top())
        p.scale(r.width(), r.height())
        p.drawRect(0, 0, 1, 1)

    def addScaleHandle(self, *args, **kwargs):
        if self.resizable:
            super().addScaleHandle(*args, **kwargs)
