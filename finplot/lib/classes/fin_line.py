import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from .. import definitions as defs
from .. import functions

class FinLine(pg.GraphicsObject):
    def __init__(self, points, pen):
        super().__init__()
        self.points = points
        self.pen = pen

    def paint(self, p, *args):
        p.setPen(self.pen)
        p.drawPath(self.shape())

    def shape(self):
        p = QtGui.QPainterPath()
        p.moveTo(*self.points[0])
        p.lineTo(*self.points[1])
        return p

    def boundingRect(self):
        return self.shape().boundingRect()
