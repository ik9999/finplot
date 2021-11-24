# -*- coding: utf-8 -*-
'''
Financial data plotter with better defaults, api, behavior and performance than
mpl_finance and plotly.

Lines up your time-series with a shared X-axis; ideal for volume, RSI, etc.

Zoom does something similar to what you'd normally expect for financial data,
where the Y-axis is auto-scaled to highest high and lowest low in the active
region.
'''

from functools import partial, partialmethod
from math import ceil, floor, fmod
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from .lib import *

from .lib import functions
from .lib import definitions as defs

def create_plot(title='Finance Plot', rows=1, init_zoom_periods=1e10, maximize=True, yscale='linear'):
    pg.setConfigOptions(foreground=defs.foreground, background=defs.background)
    win = FinWindow(title)
    # normally first graph is of higher significance, so enlarge
    win.ci.layout.setRowStretchFactor(0, defs.top_graph_scale)
    win.show_maximized = maximize
    ax0 = axs = create_plot_widget(master=win, rows=rows, init_zoom_periods=init_zoom_periods, yscale=yscale)
    axs = axs if type(axs) in (tuple,list) else [axs]
    for ax in axs:
        win.addItem(ax)
        win.nextRow()
    return ax0


def create_plot_widget(master, rows=1, init_zoom_periods=1e10, yscale='linear'):
    pg.setConfigOptions(foreground=defs.foreground, background=defs.background)
    if master not in defs.windows:
        defs.windows.append(master)
    axs = []
    prev_ax = None
    for n in range(rows):
        ysc = yscale[n] if type(yscale) in (list,tuple) else yscale
        ysc = YScale(ysc, 1)
        viewbox = FinViewBox(master, init_steps=init_zoom_periods, yscale=ysc, v_zoom_scale=1-defs.y_pad, enableMenu=False)
        ax = prev_ax = functions._add_timestamp_plot(master=master, prev_ax=prev_ax, viewbox=viewbox, index=n, yscale=ysc)
        if axs:
            ax.setXLink(axs[0].vb)
        else:
            viewbox.setFocus()
        axs += [ax]
    if isinstance(master, pg.GraphicsLayoutWidget):
        proxy = pg.SignalProxy(master.scene().sigMouseMoved, rateLimit=144, slot=partial(functions._mouse_moved, master))
    else:
        proxy = []
        for ax in axs:
            proxy += [pg.SignalProxy(ax.ax_widget.scene().sigMouseMoved, rateLimit=144, slot=partial(functions._mouse_moved, master))]
    defs.master_data[master] = dict(proxymm=proxy, last_mouse_evs=None, last_mouse_y=0)
    defs.last_ax = axs[0]
    return axs[0] if len(axs) == 1 else axs


def close():
    for win in defs.windows:
        try:
            win.close()
        except Exception as e:
            print('Window closing error:', type(e), e)
    defs.windows.clear()
    defs.overlay_axs.clear()
    functions._clear_timers()
    defs.sounds.clear()
    defs.master_data.clear()
    defs.last_ax = None


def price_colorfilter(item, datasrc, df):
    opencol = df.columns[1]
    closecol = df.columns[2]
    is_up = df[opencol] <= df[closecol] # open lower than close = goes up
    yield item.rowcolors('bull') + [df.loc[is_up, :]]
    yield item.rowcolors('bear') + [df.loc[~is_up, :]]


def volume_colorfilter(item, datasrc, df):
    opencol = df.columns[3]
    closecol = df.columns[4]
    is_up = df[opencol] <= df[closecol] # open lower than close = goes up
    yield item.rowcolors('bull') + [df.loc[is_up, :]]
    yield item.rowcolors('bear') + [df.loc[~is_up, :]]


def strength_colorfilter(item, datasrc, df):
    opencol = df.columns[1]
    closecol = df.columns[2]
    startcol = df.columns[3]
    endcol = df.columns[4]
    is_up = df[opencol] <= df[closecol] # open lower than close = goes up
    is_strong = df[startcol] <= df[endcol]
    yield item.rowcolors('bull') + [df.loc[is_up&is_strong, :]]
    yield item.rowcolors('weak_bull') + [df.loc[is_up&(~is_strong), :]]
    yield item.rowcolors('weak_bear') + [df.loc[(~is_up)&is_strong, :]]
    yield item.rowcolors('bear') + [df.loc[(~is_up)&(~is_strong), :]]


def volume_colorfilter_section(sections=[]):
    '''The sections argument is a (starting_index, color_name) array.'''
    def _colorfilter(sections, item, datasrc, df):
        if not sections:
            return volume_colorfilter(item, datasrc, df)
        for (i0,colname),(i1,_) in zip(sections, sections[1:]+[(None,'neutral')]):
            rows = df.iloc[i0:i1, :]
            yield item.rowcolors(colname) + [rows]
    return partial(_colorfilter, sections)


def horizvol_colorfilter(sections=[]):
    '''The sections argument is a (starting_index, color_name) array.'''
    def _colorfilter(sections, item, datasrc, data):
        if not sections:
            yield item.rowcolors('neutral') + [data]
        for (i0,colname),(i1,_) in zip(sections, sections[1:]+[(None,'neutral')]):
            rows = data[:, i0:i1]
            yield item.rowcolors(colname) + [rows]
    return partial(_colorfilter, sections)


def candlestick_ochl(datasrc, draw_body=True, draw_shadow=True, candle_width=0.6, ax=None, colorfunc=price_colorfilter):
    ax = functions._create_plot(ax=ax, maximize=False)
    datasrc = functions._create_datasrc(ax, datasrc)
    datasrc.scale_cols = [3,4] # only hi+lo scales
    functions._set_datasrc(ax, datasrc)
    item = CandlestickItem(ax=ax, datasrc=datasrc, draw_body=draw_body, draw_shadow=draw_shadow, candle_width=candle_width, colorfunc=colorfunc)
    functions._update_significants(ax, datasrc, force=True)
    item.update_data = partial(functions._update_data, None, None, item)
    item.update_gfx = partial(functions._update_gfx, item)
    ax.addItem(item)
    return item


def renko(x, y=None, bins=None, step=None, ax=None, colorfunc=price_colorfilter):
    ax = functions._create_plot(ax=ax, maximize=False)
    datasrc = functions._create_datasrc(ax, x, y)
    origdf = datasrc.df
    adj = functions._adjust_renko_log_datasrc if ax.vb.yscale.scaletype == 'log' else functions._adjust_renko_datasrc
    step_adjust_renko_datasrc = partial(adj, bins, step)
    step_adjust_renko_datasrc(datasrc)
    ax.decouple()
    item = candlestick_ochl(datasrc, draw_shadow=False, candle_width=1, ax=ax, colorfunc=colorfunc)
    item.colors['bull_body'] = item.colors['bull_frame']
    item.update_data = partial(functions._update_data, None, step_adjust_renko_datasrc, item)
    item.update_gfx = partial(functions._update_gfx, item)
    defs.epoch_period = (origdf.iloc[1,0] - origdf.iloc[0,0]) // int(1e9)
    return item


def volume_ocv(datasrc, candle_width=0.8, ax=None, colorfunc=volume_colorfilter):
    ax = functions._create_plot(ax=ax, maximize=False)
    datasrc = functions._create_datasrc(ax, datasrc)
    functions._adjust_volume_datasrc(datasrc)
    functions._set_datasrc(ax, datasrc)
    item = CandlestickItem(ax=ax, datasrc=datasrc, draw_body=True, draw_shadow=False, candle_width=candle_width, colorfunc=colorfunc)
    functions._update_significants(ax, datasrc, force=True)
    item.colors['bull_body'] = item.colors['bull_frame']
    if colorfunc == volume_colorfilter: # assume normal volume plot
        item.colors['bull_frame'] = defs.volume_bull_color
        item.colors['bull_body']  = defs.volume_bull_body_color
        item.colors['bear_frame'] = defs.volume_bear_color
        item.colors['bear_body']  = defs.volume_bear_color
        ax.vb.v_zoom_baseline = 0
    else:
        item.colors['weak_bull_frame'] = functions.brighten(defs.volume_bull_color, 1.2)
        item.colors['weak_bull_body']  = functions.brighten(defs.volume_bull_color, 1.2)
    item.update_data = partial(functions._update_data, None, functions._adjust_volume_datasrc, item)
    item.update_gfx = partial(functions._update_gfx, item)
    ax.addItem(item)
    item.setZValue(-20)
    return item


def horiz_time_volume(datasrc, ax=None, **kwargs):
    '''Draws multiple fixed horizontal volumes. The input format is:
       [[time0, [(price0,volume0),(price1,volume1),...]], ...]

       This chart needs to be plot last, so it knows if it controls
       what time periods are shown, or if its using time already in
       place by another plot.'''
    # update handling default if necessary
    if defs.max_zoom_points > 15:
        defs.max_zoom_points = 4
    if defs.right_margin_candles > 3:
        defs.right_margin_candles = 1

    ax = functions._create_plot(ax=ax, maximize=False)
    datasrc = functions._preadjust_horiz_datasrc(datasrc)
    datasrc = functions._create_datasrc(ax, datasrc)
    functions._adjust_horiz_datasrc(datasrc)
    if ax.vb.datasrc is not None:
        datasrc.standalone = True # only force standalone if there is something on our charts already
    datasrc.scale_cols = [datasrc.col_data_offset, len(datasrc.df.columns)-2] # first and last price columns
    datasrc.pre_update = lambda df: df.loc[:, :df.columns[0]] # throw away previous data
    datasrc.post_update = lambda df: df.dropna(how='all') # kill all-NaNs
    functions._set_datasrc(ax, datasrc)
    item = HorizontalTimeVolumeItem(ax=ax, datasrc=datasrc, **kwargs)
    item.update_data = partial(functions._update_data, functions._preadjust_horiz_datasrc, functions._adjust_horiz_datasrc, item)
    item.update_gfx = partial(functions._update_gfx, item)
    item.setZValue(-10)
    ax.addItem(item)
    return item


def heatmap(datasrc, ax=None, **kwargs):
    '''Expensive function. Only use on small data sets. See HeatmapItem for kwargs. Input datasrc
       has x (time) in index or first column, y (price) as column names, and intensity (color) as
       cell values.'''
    ax = functions._create_plot(ax=ax, maximize=False)
    if ax.vb.v_zoom_scale >= 0.9:
        ax.vb.v_zoom_scale = 0.6
    datasrc = functions._create_datasrc(ax, datasrc)
    datasrc.scale_cols = [] # doesn't scale
    functions._set_datasrc(ax, datasrc)
    item = HeatmapItem(ax=ax, datasrc=datasrc, **kwargs)
    item.update_data = partial(functions._update_data, None, None, item)
    item.update_gfx = partial(functions._update_gfx, item)
    item.setZValue(-30)
    ax.addItem(item)
    if ax.vb.datasrc is not None and not ax.vb.datasrc.timebased(): # manual zoom update
        ax.setXLink(None)
        if ax.prev_ax:
            ax.prev_ax.set_visible(xaxis=True)
        df = ax.vb.datasrc.df
        prices = df.columns[ax.vb.datasrc.col_data_offset:item.col_data_end]
        delta_price = abs(prices[0] - prices[1])
        ax.vb.set_range(0, min(df.columns[1:]), len(df), max(df.columns[1:])+delta_price)
    return item


def bar(x, y=None, width=0.8, ax=None, colorfunc=strength_colorfilter, **kwargs):
    '''Bar plots are decoupled. Use volume_ocv() if you want a bar plot which relates to other time plots.'''
    defs.right_margin_candles = 0
    defs.max_zoom_points = min(defs.max_zoom_points, 8)
    ax = functions._create_plot(ax=ax, maximize=False)
    ax.decouple()
    datasrc = functions._create_datasrc(ax, x, y)
    functions._adjust_bar_datasrc(datasrc, order_cols=False) # don't rearrange columns, done for us in volume_ocv()
    item = volume_ocv(datasrc, candle_width=width, ax=ax, colorfunc=colorfunc)
    item.update_data = partial(functions._update_data, None, functions._adjust_bar_datasrc, item)
    item.update_gfx = partial(functions._update_gfx, item)
    ax.vb.pre_process_data()
    if ax.vb.y_min >= 0:
        ax.vb.v_zoom_baseline = 0
    return item


def hist(x, bins, ax=None, **kwargs):
    hist_data = pd.cut(x, bins=bins).value_counts()
    data = [(i.mid,0,hist_data.loc[i],hist_data.loc[i]) for i in sorted(hist_data.index)]
    df = pd.DataFrame(data, columns=['x','_op_','_cl_','bin'])
    df.set_index('x', inplace=True)
    item = bar(df, ax=ax)
    del item.update_data
    return item


def plot(x, y=None, color=None, width=1, ax=None, style=None, legend=None, zoomscale=True, **kwargs):
    ax = functions._create_plot(ax=ax, maximize=False)
    used_color = functions._get_color(ax, style, color)
    datasrc = functions._create_datasrc(ax, x, y)
    if not zoomscale:
        datasrc.scale_cols = []
    functions._set_datasrc(ax, datasrc)
    if legend is not None:
        functions._create_legend(ax)
    x = datasrc.x if not ax.vb.x_indexed else datasrc.index
    y = datasrc.y / ax.vb.yscale.scalef
    if ax.vb.yscale.scaletype == 'log':
        y = y + defs.log_plot_offset
    if style is None or any(ch in style for ch in '-_.'):
        connect_dots = 'finite' # same as matplotlib; use datasrc.standalone=True if you want to keep separate intervals on a plot
        item = ax.plot(x, y, pen=functions._makepen(color=used_color, style=style, width=width), name=legend, connect=connect_dots)
        item.setZValue(5)
    else:
        symbol = {'v':'t', '^':'t1', '>':'t2', '<':'t3'}.get(style, style) # translate some similar styles
        yfilter = y.notnull()
        ser = y.loc[yfilter]
        x = x.loc[yfilter].values if hasattr(x, 'loc') else x[yfilter]
        item = ax.plot(x, ser.values, pen=None, symbol=symbol, symbolPen=None, symbolSize=7*width, symbolBrush=pg.mkBrush(used_color), name=legend)
        if width < 1:
            item.opts['antialias'] = True
        item.scatter._dopaint = item.scatter.paint
        item.scatter.paint = partial(functions._paint_scatter, item.scatter)
        # optimize (when having large number of points) by ignoring scatter click detection
        _dummy_mouse_click = lambda ev: 0
        item.scatter.mouseClickEvent = _dummy_mouse_click
        item.setZValue(10)
    item.opts['handed_color'] = color
    item.ax = ax
    item.datasrc = datasrc
    functions._update_significants(ax, datasrc, force=False)
    item.update_data = partial(functions._update_data, None, None, item)
    item.update_gfx = partial(functions._update_gfx, item)
    # add legend to main ax, not to overlay
    axm = ax.vb.master_viewbox.parent() if ax.vb.master_viewbox else ax
    if axm.legend is not None:
        if legend and axm != ax:
            axm.legend.addItem(item, name=legend)
        for _,label in axm.legend.items:
            if label.text == legend:
                label.setAttr('justify', 'left')
                label.setText(label.text, color=defs.legend_text_color)
    return item


def labels(x, y=None, labels=None, color=None, ax=None, anchor=(0.5,1)):
    ax = functions._create_plot(ax=ax, maximize=False)
    used_color = functions._get_color(ax, '?', color)
    datasrc = functions._create_datasrc(ax, x, y, labels)
    datasrc.scale_cols = [] # don't use this for scaling
    functions._set_datasrc(ax, datasrc)
    item = ScatterLabelItem(ax=ax, datasrc=datasrc, color=used_color, anchor=anchor)
    functions._update_significants(ax, datasrc, force=False)
    item.update_data = partial(functions._update_data, None, None, item)
    item.update_gfx = partial(functions._update_gfx, item)
    ax.addItem(item)
    if ax.vb.v_zoom_scale > 0.9: # adjust to make hi/lo text fit
        ax.vb.v_zoom_scale = 0.9
    return item


def add_legend(text, ax=None):
    ax = functions._create_plot(ax=ax, maximize=False)
    functions._create_legend(ax)
    row = ax.legend.layout.rowCount()
    label = pg.LabelItem(text, color=defs.legend_text_color, justify='left')
    ax.legend.layout.addItem(label, row, 0, 1, 2)
    return label


def fill_between(plot0, plot1, color=None):
    used_color = functions.brighten(functions._get_color(plot0.ax, None, color), 1.3)
    item = pg.FillBetweenItem(plot0, plot1, brush=pg.mkBrush(used_color))
    item.ax = plot0.ax
    item.setZValue(-40)
    item.ax.addItem(item)
    return item


def set_x_pos(xmin, xmax, ax=None):
    ax = functions._create_plot(ax=ax, maximize=False)
    xidx0,xidx1 = _pdtime2index(ax, pd.Series([xmin, xmax]))
    ax.vb.update_y_zoom(xidx0, xidx1)
    functions._repaint_candles()


def set_y_range(ymin, ymax, ax=None):
    ax = functions._create_plot(ax=ax, maximize=False)
    ax.setLimits(yMin=ymin, yMax=ymax)
    ax.vb.v_autozoom = False
    ax.vb.set_range(None, ymin, None, ymax)


def set_y_scale(yscale='linear', ax=None):
    ax = functions._create_plot(ax=ax, maximize=False)
    ax.setLogMode(y=(yscale=='log'))
    ax.vb.yscale = YScale(yscale, ax.vb.yscale.scalef)


def add_band(y0, y1, color=defs.band_color, ax=None):
    ax = functions._create_plot(ax=ax, maximize=False)
    color = functions._get_color(ax, None, color)
    ix = ax.vb.yscale.invxform
    lr = pg.LinearRegionItem([ix(y0),ix(y1)], orientation=pg.LinearRegionItem.Horizontal, brush=pg.mkBrush(color), movable=False)
    lr.lines[0].setPen(pg.mkPen(None))
    lr.lines[1].setPen(pg.mkPen(None))
    lr.setZValue(-50)
    lr.ax = ax
    ax.addItem(lr)
    return lr


def add_rect(p0, p1, color=defs.band_color, interactive=False, ax=None):
    ax = functions._create_plot(ax=ax, maximize=False)
    x_pts = _pdtime2index(ax, pd.Series([p0[0], p1[0]]))
    ix = ax.vb.yscale.invxform
    y0,y1 = sorted([p0[1], p1[1]])
    pos  = (x_pts[0], ix(y0))
    size = (x_pts[1]-pos[0], ix(y1-y0))
    rect = FinRect(ax=ax, brush=pg.mkBrush(color), pos=pos, size=size, movable=interactive, resizable=interactive, rotatable=False)
    rect.setZValue(-40)
    if interactive:
        ax.vb.rois.append(rect)
    rect.ax = ax
    ax.addItem(rect)
    return rect


def add_line(p0, p1, color=defs.draw_line_color, width=1, style=None, interactive=False, ax=None):
    ax = functions._create_plot(ax=ax, maximize=False)
    used_color = functions._get_color(ax, style, color)
    pen = functions._makepen(color=used_color, style=style, width=width)
    x_pts = _pdtime2index(ax, pd.Series([p0[0], p1[0]]))
    ix = ax.vb.yscale.invxform
    pts = [(x_pts[0], ix(p0[1])), (x_pts[1], ix(p1[1]))]
    if interactive:
        line = FinPolyLine(ax.vb, pts, closed=False, pen=pen, movable=False)
        ax.vb.rois.append(line)
    else:
        line = FinLine(pts, pen=pen)
    line.ax = ax
    ax.addItem(line)
    return line


def add_text(pos, s, color=defs.draw_line_color, anchor=(0,0), ax=None):
    ax = functions._create_plot(ax=ax, maximize=False)
    color = functions._get_color(ax, None, color)
    text = pg.TextItem(s, color=color, anchor=anchor)
    x = pos[0]
    if ax.vb.datasrc is not None:
        x = _pdtime2index(ax, pd.Series([pos[0]]))[0]
    y = ax.vb.yscale.invxform(pos[1])
    text.setPos(x, y)
    text.setZValue(50)
    text.ax = ax
    ax.addItem(text, ignoreBounds=True)
    return text


def remove_line(line):
    print('remove_line() is deprecated, use remove_primitive() instead')
    remove_primitive(line)


def remove_text(text):
    print('remove_text() is deprecated, use remove_primitive() instead')
    remove_primitive(text)


def remove_primitive(primitive):
    ax = primitive.ax
    ax.removeItem(primitive)
    if primitive in ax.vb.rois:
        ax.vb.rois.remove(primitive)
    if hasattr(primitive, 'texts'):
        for txt in primitive.texts:
            ax.vb.removeItem(txt)


def set_time_inspector(inspector, ax=None, when='click'):
    '''Callback when clicked like so: inspector(x, y).'''
    ax = ax if ax else defs.last_ax
    win = ax.vb.win
    if when == 'hover':
        win.proxy_hover = pg.SignalProxy(win.scene().sigMouseMoved, rateLimit=15, slot=partial(functions._inspect_pos, ax, inspector))
    elif when in ('dclick', 'double-click'):
        win.proxy_dclick = pg.SignalProxy(win.scene().sigMouseClicked, slot=partial(functions._inspect_clicked, ax, inspector, True))
    else:
        win.proxy_click = pg.SignalProxy(win.scene().sigMouseClicked, slot=partial(functions._inspect_clicked, ax, inspector, False))


def add_crosshair_info(infofunc, ax=None):
    '''Callback when crosshair updated like so: info(ax,x,y,xtext,ytext); the info()
       callback must return two values: xtext and ytext.'''
    ax = functions._create_plot(ax=ax, maximize=False)
    ax.crosshair.infos.append(infofunc)


def timer_callback(update_func, seconds, single_shot=False):
    timer = QtCore.QTimer()
    timer.timeout.connect(update_func)
    if single_shot:
        timer.setSingleShot(True)
    timer.start(int(seconds*1000))
    defs.timers.append(timer)
    return timer


def autoviewrestore(enable=True):
    '''Restor functionality saves view zoom coordinates when closing a window, and
       load them when creating the plot (with the same name) again.'''
    defs.viewrestore = enable


def refresh():
    for win in defs.windows:
        vbs = [ax.vb for ax in win.axs] + [ax.vb for ax in defs.overlay_axs if ax.vb.win==win]
        for vb in vbs:
            vb.pre_process_data()
        if defs.viewrestore:
            if functions._loadwindata(win):
                continue
        functions._set_max_zoom(vbs)
        for vb in vbs:
            datasrc = vb.datasrc_or_standalone
            if datasrc and (vb.linkedView(0) is None or vb.linkedView(0).datasrc is None or vb.master_viewbox):
                vb.update_y_zoom(datasrc.init_x0, datasrc.init_x1)
    functions._repaint_candles()
    for win in defs.windows:
        functions._mouse_moved(win, None)


def show(qt_exec=True):
    refresh()
    for win in defs.windows:
        if isinstance(win, FinWindow) or qt_exec:
            if win.show_maximized:
                win.showMaximized()
            else:
                win.show()
    if defs.windows and qt_exec:
        defs.app = QtGui.QApplication.instance()
        defs.app.exec_()
        defs.windows.clear()
        defs.overlay_axs.clear()
        functions._clear_timers()
        defs.sounds.clear()
        defs.master_data.clear()
        defs.last_ax = None


def play_sound(filename):
    if filename not in defs.sounds:
        from PyQt5.QtMultimedia import QSound
        defs.sounds[filename] = QSound(filename) # disallow gc
    s = defs.sounds[filename]
    s.play()


def screenshot(file, fmt='png'):
    if functions._internal_windows_only() and not defs.app:
        print('ERROR: screenshot must be callbacked from e.g. timer_callback()')
        return False
    try:
        buffer = QtCore.QBuffer()
        defs.app.primaryScreen().grabWindow(defs.windows[0].winId()).save(buffer, fmt)
        file.write(buffer.data())
        return True
    except Exception as e:
        print('Screenshot error:', type(e), e)
    return False


try:
    qtver = '%d.%d' % (QtCore.QT_VERSION//256//256, QtCore.QT_VERSION//256%256)
    if qtver not in ('5.9', '5.13') and [int(i) for i in pg.__version__.split('.')] <= [0,11,0]:
        print('WARNING: your version of Qt may not plot curves containing NaNs and is not recommended.')
        print('See https://github.com/pyqtgraph/pyqtgraph/issues/1057')
except:
    pass


# default to black-on-white
pg.widgets.GraphicsView.GraphicsView.wheelEvent = partialmethod(
    functions._wheel_event_wrapper, pg.widgets.GraphicsView.GraphicsView.wheelEvent
)
# use finplot instead of matplotlib
pd.set_option('plotting.backend', 'finplot.pdplot')
# pick up win resolution
try:
    import ctypes
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    defs.lod_candles = int(user32.GetSystemMetrics(0) * 1.6)
    defs.candle_shadow_width = int(user32.GetSystemMetrics(0) // 2100 + 1) # 2560 and resolutions above -> wider shadows
except:
    pass


if False: # performance measurement code
    import time, sys
    def self_timecall(self, pname, fname, func, *args, **kwargs):
        ## print('self_timecall', pname, fname)
        t0 = time.perf_counter()
        r = func(self, *args, **kwargs)
        t1 = time.perf_counter()
        print('%s.%s: %f' % (pname, fname, t1-t0))
        return r
    def timecall(fname, func, *args, **kwargs):
        ## print('timecall', fname)
        t0 = time.perf_counter()
        r = func(*args, **kwargs)
        t1 = time.perf_counter()
        print('%s: %f' % (fname, t1-t0))
        return r
    def wrappable(fn, f):
        try:    return callable(f) and str(f.__module__) == 'finplot'
        except: return False
    m = sys.modules['finplot']
    for fname in dir(m):
        func = getattr(m, fname)
        if wrappable(fname, func):
            for fname2 in dir(func):
                func2 = getattr(func, fname2)
                if wrappable(fname2, func2):
                    print(fname, str(type(func)), '->', fname2, str(type(func2)))
                    setattr(func, fname2, partialmethod(self_timecall, fname, fname2, func2))
            setattr(m, fname, partial(timecall, fname, func))
