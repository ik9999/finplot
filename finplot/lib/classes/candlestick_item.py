import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from .. import definitions as defs
from .. import functions
from .fin_plot_item import FinPlotItem

class CandlestickItem(FinPlotItem):
    def __init__(self, ax, datasrc, draw_body, draw_shadow, candle_width, colorfunc):
        self.colors = dict(bull_shadow      = defs.candle_bull_color,
                           bull_frame       = defs.candle_bull_color,
                           bull_body        = defs.candle_bull_body_color,
                           bear_shadow      = defs.candle_bear_color,
                           bear_frame       = defs.candle_bear_color,
                           bear_body        = defs.candle_bear_color,
                           weak_bull_shadow = functions.brighten(defs.candle_bull_color, 1.2),
                           weak_bull_frame  = functions.brighten(defs.candle_bull_color, 1.2),
                           weak_bull_body   = functions.brighten(defs.candle_bull_color, 1.2),
                           weak_bear_shadow = functions.brighten(defs.candle_bear_color, 1.5),
                           weak_bear_frame  = functions.brighten(defs.candle_bear_color, 1.5),
                           weak_bear_body   = functions.brighten(defs.candle_bear_color, 1.5))
        self.draw_body = draw_body
        self.draw_shadow = draw_shadow
        self.candle_width = candle_width
        self.shadow_width = defs.candle_shadow_width
        self.colorfunc = colorfunc
        self.x_offset = 0
        super().__init__(ax, datasrc, lod=True)

    def generate_picture(self, boundingRect):
        w = self.candle_width
        w2 = w * 0.5
        left,right = boundingRect.left(), boundingRect.right()
        p = self.painter
        df,origlen = self.datasrc.rows(5, left, right, yscale=self.ax.vb.yscale)
        drawing_many_shadows = self.draw_shadow and origlen > defs.lod_candles*2//3
        for shadow,frame,body,df_rows in self.colorfunc(self, self.datasrc, df):
            idxs = df_rows.index
            rows = df_rows.values
            if self.x_offset:
                idxs += self.x_offset
            if self.draw_shadow:
                p.setPen(pg.mkPen(shadow, width=self.shadow_width))
                for x,(t,open,close,high,low) in zip(idxs, rows):
                    if high > low:
                        p.drawLine(QtCore.QPointF(x, low), QtCore.QPointF(x, high))
            if self.draw_body and not drawing_many_shadows: # settle with only drawing shadows if too much detail
                p.setPen(pg.mkPen(frame))
                p.setBrush(pg.mkBrush(body))
                for x,(t,open,close,high,low) in zip(idxs, rows):
                    p.drawRect(QtCore.QRectF(x-w2, open, w, close-open))

    def rowcolors(self, prefix):
        return [self.colors[prefix+'_shadow'], self.colors[prefix+'_frame'], self.colors[prefix+'_body']]
