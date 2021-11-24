from .. import definitions as defs
from .. import functions
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

class FinCrossHair:
    def __init__(self, ax, color):
        self.ax = ax
        self.x = 0
        self.y = 0
        self.clamp_x = 0
        self.clamp_y = 0
        self.infos = []
        pen = pg.mkPen(color=color, style=QtCore.Qt.CustomDashLine, dash=[7, 7])
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        self.xtext = pg.TextItem(color=color, anchor=(0,1))
        self.ytext = pg.TextItem(color=color, anchor=(0,0))
        self.vline.setZValue(50)
        self.hline.setZValue(50)
        self.xtext.setZValue(50)
        self.ytext.setZValue(50)
        self.show()

    def update(self, point=None):
        if point is not None:
            self.x,self.y = x,y = point.x(),point.y()
        else:
            x,y = self.x,self.y
        x,y = functions._clamp_xy(self.ax, x,y)
        if x == self.clamp_x and y == self.clamp_y:
            return
        self.clamp_x,self.clamp_y = x,y
        self.vline.setPos(x)
        self.hline.setPos(y)
        self.xtext.setPos(x, y)
        self.ytext.setPos(x, y)
        rng = self.ax.vb.y_max - self.ax.vb.y_min
        rngmax = abs(self.ax.vb.y_min) + rng # any approximation is fine
        sd,se = (self.ax.significant_decimals,self.ax.significant_eps) if defs.clamp_grid else (defs.significant_decimals,defs.significant_eps)
        timebased = False
        if self.ax.vb.x_indexed:
            xtext,timebased = functions._x2local_t(self.ax.vb.datasrc, x)
        else:
            xtext = functions._round_to_significant(rng, rngmax, x, sd, se)
        linear_y = y
        y = self.ax.vb.yscale.xform(y)
        ytext = functions._round_to_significant(rng, rngmax, y, sd, se)
        if not timebased:
            if xtext:
                xtext = 'x ' + xtext
            ytext = 'y ' + ytext
        far_right = self.ax.viewRect().x() + self.ax.viewRect().width()*0.9
        far_bottom = self.ax.viewRect().y() + self.ax.viewRect().height()*0.1
        close2right = x > far_right
        close2bottom = linear_y < far_bottom
        try:
            for info in self.infos:
                xtext,ytext = info(x,y,xtext,ytext)
        except Exception as e:
            print('Crosshair error:', type(e), e)
        space = '      '
        if close2right:
            xtext = xtext + space
            ytext = ytext + space
            xanchor = [1,1]
            yanchor = [1,0]
        else:
            xtext = space + xtext
            ytext = space + ytext
            xanchor = [0,1]
            yanchor = [0,0]
        if close2bottom:
            ytext = ytext + space
            yanchor = [1,1]
            if close2right:
                xanchor = [1,2]
        self.xtext.setAnchor(xanchor)
        self.ytext.setAnchor(yanchor)
        self.xtext.setText(xtext)
        self.ytext.setText(ytext)

    def show(self):
        self.ax.addItem(self.vline, ignoreBounds=True)
        self.ax.addItem(self.hline, ignoreBounds=True)
        self.ax.addItem(self.xtext, ignoreBounds=True)
        self.ax.addItem(self.ytext, ignoreBounds=True)

    def hide(self):
        self.ax.removeItem(self.xtext)
        self.ax.removeItem(self.ytext)
        self.ax.removeItem(self.vline)
        self.ax.removeItem(self.hline)


