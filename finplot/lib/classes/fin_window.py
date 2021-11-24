import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from .. import definitions as defs
from .. import functions

class FinWindow(pg.GraphicsLayoutWidget):
    def __init__(self, title, **kwargs):
        self.title = title
        pg.mkQApp()
        super().__init__(**kwargs)
        self.setWindowTitle(title)
        self.setGeometry(defs.winx, defs.winy, defs.winw, defs.winh)
        defs.winx += 40
        defs.winy += 40
        self.centralWidget.installEventFilter(self)
        self.ci.setContentsMargins(0, 0, 0, 0)
        self.ci.setSpacing(-1)
        self.closing = False

    @property
    def axs(self):
        return [ax for ax in self.ci.items if isinstance(ax, pg.PlotItem)]

    def close(self):
        self.closing = True
        functions._savewindata(self)
        functions._clear_timers()
        return super().close()

    def eventFilter(self, obj, ev):
        if ev.type()== QtCore.QEvent.WindowDeactivate:
            functions._savewindata(self)
        return False

    def leaveEvent(self, ev):
        if not self.closing:
            super().leaveEvent(ev)

