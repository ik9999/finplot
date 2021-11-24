import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from .. import definitions as defs
from .. import functions
from .y_scale import YScale

class FinViewBox(pg.ViewBox):
    def __init__(self, win, init_steps=300, yscale=YScale('linear', 1), v_zoom_scale=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.win = win
        self.init_steps = init_steps
        self.yscale = yscale
        self.v_zoom_scale = v_zoom_scale
        self.master_viewbox = None
        self.rois = []
        self.win._isMouseLeftDrag = False
        self.reset()

    def reset(self):
        self.v_zoom_baseline = 0.5
        self.v_autozoom = True
        self.max_zoom_points_f = 1
        self.y_max = 1000
        self.y_min = 0
        self.y_positive = True
        self.x_indexed = True
        self.force_range_update = 0
        while self.rois:
            self.remove_last_roi()
        self.draw_line = None
        self.drawing = False
        self.standalones = set()
        self.updating_linked = False
        self.set_datasrc(None)
        self.setMouseEnabled(x=True, y=False)
        self.setRange(QtCore.QRectF(pg.Point(0, 0), pg.Point(1, 1)))

    def set_datasrc(self, datasrc):
        self.datasrc = datasrc
        if not self.datasrc:
            return
        datasrc.update_init_x(self.init_steps)

    def pre_process_data(self):
        if self.datasrc and self.datasrc.scale_cols:
            df = self.datasrc.df.iloc[:, self.datasrc.scale_cols]
            self.y_max = df.max().max()
            self.y_min = df.min().min()
            if self.y_min <= 0:
                self.y_positive = False

    @property
    def datasrc_or_standalone(self):
        ds = self.datasrc
        if not ds and self.standalones:
            ds = next(iter(self.standalones))
        return ds

    def wheelEvent(self, ev, axis=None):
        if self.master_viewbox:
            return self.master_viewbox.wheelEvent(ev, axis=axis)
        if ev.modifiers() == QtCore.Qt.ControlModifier:
            scale_fact = 1
            self.v_zoom_scale /= 1.02 ** (ev.delta() * self.state['wheelScaleFactor'])
        else:
            scale_fact = 1.02 ** (ev.delta() * self.state['wheelScaleFactor'])
        vr = self.targetRect()
        center = self.mapToView(ev.pos())
        if (center.x()-vr.left())/vr.width() < 0.05: # zoom to far left => all the way left
            center = pg.Point(vr.left(), center.y())
        elif (center.x()-vr.left())/vr.width() > 0.95: # zoom to far right => all the way right
            center = pg.Point(vr.right(), center.y())
        self.zoom_rect(vr, scale_fact, center)
        # update crosshair
        functions._mouse_moved(self.win, None)
        ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        axis = 0 # don't constrain drag direction
        if self.master_viewbox:
            return self.master_viewbox.mouseDragEvent(ev, axis=axis)
        if not self.datasrc:
            return
        if ev.button() == QtCore.Qt.LeftButton:
            self.mouseLeftDrag(ev, axis)
        elif ev.button() == QtCore.Qt.MiddleButton:
            self.mouseMiddleDrag(ev, axis)
        elif ev.button() == QtCore.Qt.RightButton:
            self.mouseRightDrag(ev, axis)
        else:
            super().mouseDragEvent(ev, axis)

    def mouseLeftDrag(self, ev, axis):
        '''Ctrl+LButton draw lines.'''
        if ev.modifiers() != QtCore.Qt.ControlModifier:
            super().mouseDragEvent(ev, axis)
            if ev.isFinish():
                self.win._isMouseLeftDrag = False
            else:
                self.win._isMouseLeftDrag = True
            if ev.isFinish() or self.drawing:
                self.refresh_all_y_zoom()
            if not self.drawing:
                return
        if self.draw_line and not self.drawing:
            self.set_draw_line_color(defs.draw_done_color)
        p1 = self.mapToView(ev.pos())
        p1 = functions._clamp_point(self.parent(), p1)
        if not self.drawing:
            # add new line
            p0 = self.mapToView(ev.lastPos())
            p0 = functions._clamp_point(self.parent(), p0)
            self.draw_line = FinPolyLine(self, [p0, p1], closed=False, pen=pg.mkPen(defs.draw_line_color), movable=False)
            self.draw_line.setZValue(40)
            self.rois.append(self.draw_line)
            self.addItem(self.draw_line)
            self.drawing = True
        else:
            # draw placed point at end of poly-line
            self.draw_line.movePoint(-1, p1)
        if ev.isFinish():
            self.drawing = False
        ev.accept()

    def mouseMiddleDrag(self, ev, axis):
        '''Ctrl+MButton draw ellipses.'''
        if ev.modifiers() != QtCore.Qt.ControlModifier:
            return super().mouseDragEvent(ev, axis)
        p1 = self.mapToView(ev.pos())
        p1 = functions._clamp_point(self.parent(), p1)
        def nonzerosize(a, b):
            c = b-a
            return pg.Point(abs(c.x()) or 1, abs(c.y()) or 1e-3)
        if not self.drawing:
            # add new ellipse
            p0 = self.mapToView(ev.lastPos())
            p0 = functions._clamp_point(self.parent(), p0)
            s = nonzerosize(p0, p1)
            p0 = QtCore.QPointF(p0.x()-s.x()/2, p0.y()-s.y()/2)
            self.draw_ellipse = FinEllipse(p0, s, pen=pg.mkPen(defs.draw_line_color), movable=True)
            self.draw_ellipse.setZValue(80)
            self.rois.append(self.draw_ellipse)
            self.addItem(self.draw_ellipse)
            self.drawing = True
        else:
            c = self.draw_ellipse.pos() + self.draw_ellipse.size()*0.5
            s = nonzerosize(c, p1)
            self.draw_ellipse.setSize(s*2, update=False)
            self.draw_ellipse.setPos(c-s)
        if ev.isFinish():
            self.drawing = False
        ev.accept()

    def mouseRightDrag(self, ev, axis):
        '''RButton is box zoom. At least for now.'''
        ev.accept()
        if not ev.isFinish():
            self.updateScaleBox(ev.buttonDownPos(), ev.pos())
        else:
            self.rbScaleBox.hide()
            ax = QtCore.QRectF(pg.Point(ev.buttonDownPos(ev.button())), pg.Point(ev.pos()))
            ax = self.childGroup.mapRectFromParent(ax)
            if ax.width() < 2: # zooming this narrow is probably a mistake
                ax.adjust(-1, 0, +1, 0)
            self.showAxRect(ax)
            self.axHistoryPointer += 1
            self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]

    def mouseClickEvent(self, ev):
        if self.master_viewbox:
            return self.master_viewbox.mouseClickEvent(ev)
        if functions._mouse_clicked(self, ev):
            ev.accept()
            return
        if ev.button() != QtCore.Qt.LeftButton or ev.modifiers() != QtCore.Qt.ControlModifier or not self.draw_line:
            return super().mouseClickEvent(ev)
        # add another segment to the currently drawn line
        p = self.mapToView(ev.pos())
        p = functions._clamp_point(self.parent(), p)
        self.append_draw_segment(p)
        self.drawing = False
        ev.accept()

    def keyPressEvent(self, ev):
        if self.master_viewbox:
            return self.master_viewbox.keyPressEvent(ev)
        if functions._key_pressed(self, ev):
            ev.accept()
            return
        super().keyPressEvent(ev)

    def linkedViewChanged(self, view, axis):
        if not self.datasrc or self.updating_linked:
            return
        if view and self.datasrc and view.datasrc:
            self.updating_linked = True
            tr = self.targetRect()
            vr = view.targetRect()
            is_dirty = view.force_range_update > 0
            is_same_scale = self.datasrc.xlen == view.datasrc.xlen
            if is_same_scale: # stable zoom based on index
                if is_dirty or abs(vr.left()-tr.left()) >= 1 or abs(vr.right()-tr.right()) >= 1:
                    if is_dirty:
                        view.force_range_update -= 1
                    self.update_y_zoom(vr.left(), vr.right())
            else: # sloppy one based on time stamps
                tt0,tt1,_,_,_ = self.datasrc.hilo(tr.left(), tr.right())
                vt0,vt1,_,_,_ = view.datasrc.hilo(vr.left(), vr.right())
                period2 = self.datasrc.period_ns * 0.5
                if is_dirty or abs(vt0-tt0) >= period2 or abs(vt1-tt1) >= period2:
                    if is_dirty:
                        view.force_range_update -= 1
                    if self.parent():
                        x0,x1 = _pdtime2index(self.parent(), pd.Series([vt0,vt1]), any_end=True)
                        self.update_y_zoom(x0, x1)
            self.updating_linked = False

    def zoom_rect(self, vr, scale_fact, center):
        if not self.datasrc:
            return
        x0 = center.x() + (vr.left()-center.x()) * scale_fact
        x1 = center.x() + (vr.right()-center.x()) * scale_fact
        self.update_y_zoom(x0, x1)

    def pan_x(self, steps=None, percent=None):
        if self.datasrc is None:
            return
        if steps is None:
            steps = int(percent/100*self.targetRect().width())
        tr = self.targetRect()
        x1 = tr.right() + steps
        startx = -defs.side_margin
        endx = self.datasrc.xlen + defs.right_margin_candles - defs.side_margin
        if x1 > endx:
            x1 = endx
        x0 = x1 - tr.width()
        if x0 < startx:
            x0 = startx
            x1 = x0 + tr.width()
        self.update_y_zoom(x0, x1)

    def refresh_all_y_zoom(self):
        '''This updates Y zoom on all views, such as when a mouse drag is completed.'''
        main_vb = self
        if self.linkedView(0):
            self.force_range_update = 1 # main need to update only once to us
            main_vb = list(self.win.axs)[0].vb
        main_vb.force_range_update = len(self.win.axs)-1 # update main as many times as there are other rows
        self.update_y_zoom()
        # refresh crosshair when done
        functions._mouse_moved(self.win, None)

    def update_y_zoom(self, x0=None, x1=None):
        datasrc = self.datasrc_or_standalone
        if datasrc is None:
            return
        if x0 is None or x1 is None:
            tr = self.targetRect()
            x0 = tr.left()
            x1 = tr.right()
            if x1-x0 <= 1:
                return
        # make edges rigid
        xl = max(functions._round(x0-defs.side_margin)+defs.side_margin, -defs.side_margin)
        xr = min(functions._round(x1-defs.side_margin)+defs.side_margin, datasrc.xlen+defs.right_margin_candles-defs.side_margin)
        dxl = xl-x0
        dxr = xr-x1
        if dxl > 0:
            x1 += dxl
        if dxr < 0:
            x0 += dxr
        x0 = max(functions._round(x0-defs.side_margin)+defs.side_margin, -defs.side_margin)
        x1 = min(functions._round(x1-defs.side_margin)+defs.side_margin, datasrc.xlen+defs.right_margin_candles-defs.side_margin)
        # fetch hi-lo and set range
        _,_,hi,lo,cnt = datasrc.hilo(x0, x1)
        vr = self.viewRect()
        minlen = int((defs.max_zoom_points-0.5) * self.max_zoom_points_f + 0.51)
        if (x1-x0) < vr.width() and cnt < minlen:
            return
        if not self.v_autozoom:
            hi = vr.bottom()
            lo = vr.top()
        if self.yscale.scaletype == 'log':
            lo = max(1e-100, lo)
            rng = (hi / lo) ** (1/self.v_zoom_scale)
            rng = min(rng, 1e50) # avoid float overflow
            base = (hi*lo) ** self.v_zoom_baseline
            y0 = base / rng**self.v_zoom_baseline
            y1 = base * rng**(1-self.v_zoom_baseline)
        else:
            rng = (hi-lo) / self.v_zoom_scale
            rng = max(rng, 2e-7) # some very weird bug where high/low exponents stops rendering
            base = (hi+lo) * self.v_zoom_baseline
            y0 = base - rng*self.v_zoom_baseline
            y1 = base + rng*(1-self.v_zoom_baseline)
        if not self.x_indexed:
            x0,x1 = functions._xminmax(datasrc, x_indexed=False, extra_margin=0)
        return self.set_range(x0, y0, x1, y1)

    def set_range(self, x0, y0, x1, y1):
        if x0 is None or x1 is None:
            tr = self.targetRect()
            x0 = tr.left()
            x1 = tr.right()
        if np.isnan(y0) or np.isnan(y1):
            return
        _y0 = self.yscale.invxform(y0, verify=True)
        _y1 = self.yscale.invxform(y1, verify=True)
        self.setRange(QtCore.QRectF(pg.Point(x0, _y0), pg.Point(x1, _y1)), padding=0)
        return True

    def remove_last_roi(self):
        if self.rois:
            if not isinstance(self.rois[-1], pg.PolyLineROI):
                self.removeItem(self.rois[-1])
                self.rois = self.rois[:-1]
            else:
                h = self.rois[-1].handles[-1]['item']
                self.rois[-1].removeHandle(h)
                if not self.rois[-1].segments:
                    self.removeItem(self.rois[-1])
                    self.rois = self.rois[:-1]
                    self.draw_line = None
            if self.rois:
                if isinstance(self.rois[-1], pg.PolyLineROI):
                    self.draw_line = self.rois[-1]
                    self.set_draw_line_color(defs.draw_line_color)
            return True

    def append_draw_segment(self, p):
        h0 = self.draw_line.handles[-1]['item']
        h1 = self.draw_line.addFreeHandle(p)
        self.draw_line.addSegment(h0, h1)
        self.drawing = True

    def set_draw_line_color(self, color):
        if self.draw_line:
            pen = pg.mkPen(color)
            for segment in self.draw_line.segments:
                segment.currentPen = segment.pen = pen
                segment.update()

    def suggestPadding(self, axis):
        return 0
