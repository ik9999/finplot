import pyqtgraph as pg
import pandas as pd
from ..functions import _get_datasrc, _pdtime2index, _x2local_t
from datetime import datetime, timezone
from math import ceil, floor, fmod
from .. import definitions as defs

def _x2year(datasrc, x):
    t,hasds = _x2local_t(datasrc, x)
    return t[:4],hasds

class EpochAxisItem(pg.AxisItem):
    def __init__(self, vb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vb = vb

    def tickStrings(self, values, scale, spacing):
        if self.mode == 'num':
            return ['%g'%v for v in values]
        conv = _x2year if self.mode=='years' else _x2local_t
        strs = [conv(self.vb.datasrc, value)[0] for value in values]
        if all(s.endswith(' 00:00') for s in strs if s): # all at midnight -> round to days
            strs = [s.partition(' ')[0] for s in strs]
        return strs

    def tickValues(self, minVal, maxVal, size):
        self.mode = 'num'
        ax = self.vb.parent()
        datasrc = _get_datasrc(ax, require=False)
        if datasrc is None or not self.vb.x_indexed:
            return super().tickValues(minVal, maxVal, size)
        # calculate if we use years, days, etc.
        t0,t1,_,_,_ = datasrc.hilo(minVal, maxVal)
        t0,t1 = pd.to_datetime(t0), pd.to_datetime(t1)
        dts = (t1-t0).total_seconds()
        gfx_width = int(size)
        for mode, dtt, freq, ticklen in defs.time_splits:
            if dts > dtt:
                self.mode = mode
                desired_ticks = gfx_width / ((ticklen+2) * 10) - 1 # an approximation is fine
                if not self.vb.datasrc.is_smooth_time():
                    desired_ticks -= 1 # leave more space for unevenly spaced ticks
                desired_ticks = max(desired_ticks, 4)
                to_midnight = freq in ('YS','MS', 'W-MON', 'D')
                tz = defs.display_timezone if to_midnight else None # for shorter timeframes, timezone seems buggy
                rng = pd.date_range(t0, t1, tz=tz, normalize=to_midnight, freq=freq)
                steps = len(rng) if len(rng)&1==0 else len(rng)+1 # reduce jitter between e.g. 5<-->10 ticks for resolution close to limit
                step = int(steps/desired_ticks) or 1
                rng = rng[::step]
                if not to_midnight:
                    try:    rng = rng.round(freq=freq)
                    except: pass
                ax = self.vb.parent()
                rng = _pdtime2index(ax=ax, ts=pd.Series(rng), require_time=True)
                indices = [ceil(i) for i in rng]
                return [(0, indices)]
        return [(0,[])]

    def generateDrawSpecs(self, p):
        specs = super().generateDrawSpecs(p)
        if specs:
            if not self.style['showValues']:
                pen,p0,p1 = specs[0] # axis specs
                specs = [(_makepen('#fff0'),p0,p1)] + list(specs[1:]) # don't draw axis if hiding values
            else:
                # throw out ticks that are out of bounds
                text_specs = specs[2]
                if len(text_specs) >= 4:
                    rect,flags,text = text_specs[0]
                    if rect.left() < 0:
                        del text_specs[0]
                    rect,flags,text = text_specs[-1]
                    if rect.right() > self.geometry().width():
                        del text_specs[-1]
                # ... and those that overlap
                x = 1e6
                for i,(rect,flags,text) in reversed(list(enumerate(text_specs))):
                    if rect.right() >= x:
                        del text_specs[i]
                    else:
                        x = rect.left()
        return specs

