import pandas as pd
import numpy as np
from .. import functions
from .. import definitions as defs
from collections import OrderedDict, defaultdict

class PandasDataSource:
    '''Candle sticks: create with five columns: time, open, close, hi, lo - in that order.
       Volume bars: create with three columns: time, open, close, volume - in that order.
       For all other types, time needs to be first, usually followed by one or more Y-columns.'''
    def __init__(self, df):
        if type(df.index) == pd.DatetimeIndex or df.index[-1]>1e7 or '.RangeIndex' not in str(type(df.index)):
            df = df.reset_index()
        self.df = df.copy()
        # manage time column
        if functions._has_timecol(self.df):
            timecol = self.df.columns[0]
            dtype = str(df[timecol].dtype)
            isnum = ('int' in dtype or 'float' in dtype) and df[timecol].iloc[-1] < 1e7
            if not isnum:
                self.df[timecol] = functions._pdtime2epoch(df[timecol])
            self.standalone = functions._is_standalone(self.df[timecol])
            self.col_data_offset = 1 # no. of preceeding columns for other plots and time column
        else:
            self.standalone = False
            self.col_data_offset = 0 # no. of preceeding columns for other plots and time column
        # setup data for joining data sources and zooming
        self.scale_cols = [i for i in range(self.col_data_offset,len(self.df.columns)) if self.df.iloc[:,i].dtype!=object]
        self.cache_hilo = OrderedDict()
        self.renames = {}
        newcols = []
        for col in self.df.columns:
            oldcol = col
            while col in newcols:
                col = str(col)+'+'
            newcols.append(col)
            if oldcol != col:
                self.renames[oldcol] = col
        self.df.columns = newcols
        self.pre_update = lambda df: df
        self.post_update = lambda df: df
        self._period = None
        self._smooth_time = None
        self.is_sparse = self.df[self.df.columns[self.col_data_offset]].isnull().sum().max() > len(self.df)//2

    @property
    def period_ns(self):
        if len(self.df) <= 1:
            return 1
        if not self._period:
            timecol = self.df.columns[0]
            times = self.df[timecol].iloc[0:100]
            self._period = int(times.diff().median()) if len(times)>1 else 1
        return self._period

    @property
    def index(self):
        return self.df.index

    @property
    def x(self):
        timecol = self.df.columns[0]
        return self.df[timecol]

    @property
    def y(self):
        col = self.df.columns[self.col_data_offset]
        return self.df[col]

    @property
    def z(self):
        col = self.df.columns[self.col_data_offset+1]
        return self.df[col]

    @property
    def xlen(self):
        return len(self.df)

    def calc_significant_decimals(self):
        ser = self.z if len(self.scale_cols)>1 else self.y
        absdiff = ser.diff().abs()
        absdiff[absdiff<1e-30] = 1e30
        num = smallest_diff = absdiff.min()
        for _ in range(2):
            s = '%e' % num
            base,_,exp = s.partition('e')
            base = base.rstrip('0')
            exp = -int(exp)
            max_base_decimals = min(5, -exp+2) if exp < 0 else 3
            base_decimals = max(0, min(max_base_decimals, len(base)-2))
            decimals = exp + base_decimals
            decimals = max(0, min(10, decimals))
            if decimals <= 3:
                break
            # retry with full number, to see if we can get the number of decimals down
            num = smallest_diff + abs(ser.min())
        smallest_diff = max(10**(-decimals), smallest_diff)
        return decimals, smallest_diff

    def update_init_x(self, init_steps):
        self.init_x0, self.init_x1 = functions._xminmax(self, x_indexed=True, init_steps=init_steps)

    def closest_time(self, x):
        timecol = self.df.columns[0]
        return self.df.loc[int(x), timecol]

    def timebased(self):
        return self.df.iloc[-1,0] > 1e7

    def is_smooth_time(self):
        if self._smooth_time is None:
            # less than 1% time delta is smooth
            self._smooth_time = self.timebased() and (np.abs(np.diff(self.x.values[1:100])[1:]//(self.period_ns//1000)-1000) < 10).all()
        return self._smooth_time

    def addcols(self, datasrc):
        new_scale_cols = [c+len(self.df.columns)-datasrc.col_data_offset for c in datasrc.scale_cols]
        self.scale_cols += new_scale_cols
        orig_col_data_cnt = len(self.df.columns)
        if functions._has_timecol(datasrc.df):
            timecol = self.df.columns[0]
            df = self.df.set_index(timecol)
            timecol = timecol if timecol in datasrc.df.columns else datasrc.df.columns[0]
            newcols = datasrc.df.set_index(timecol)
        else:
            df = self.df
            newcols = datasrc.df
        cols = list(newcols.columns)
        for i,col in enumerate(cols):
            old_col = col
            while col in self.df.columns:
                cols[i] = col = str(col)+'+'
            if old_col != col:
                datasrc.renames[old_col] = col
        newcols.columns = cols
        self.df = pd.concat([df, newcols], axis=1)
        if functions._has_timecol(datasrc.df):
            self.df.reset_index(inplace=True)
        datasrc.df = self.df # they are the same now
        datasrc.init_x0 = self.init_x0
        datasrc.init_x1 = self.init_x1
        datasrc.col_data_offset = orig_col_data_cnt
        datasrc.scale_cols = new_scale_cols
        self.cache_hilo = OrderedDict()
        self._period = self._smooth_time = None
        datasrc._period = datasrc._smooth_time = None
        ldf2 = len(self.df) / 2
        self.is_sparse = self.is_sparse or self.df[self.df.columns[self.col_data_offset]].isnull().sum().max() > ldf2
        datasrc.is_sparse = datasrc.is_sparse or datasrc.df[datasrc.df.columns[datasrc.col_data_offset]].isnull().sum().max() > ldf2

    def update(self, datasrc):
        df = self.pre_update(self.df)
        orig_cols = list(df.columns)
        timecol,orig_cols = orig_cols[0],orig_cols[1:]
        df = df.set_index(timecol)
        input_df = datasrc.df.set_index(datasrc.df.columns[0])
        input_df.columns = [self.renames.get(col, col) for col in input_df.columns]
        # pad index if the input data is a sub-set
        input_df = pd.merge(input_df, df[[]], how='outer', left_index=True, right_index=True)
        for col in df.columns:
            if col not in input_df.columns:
                input_df[col] = df[col]
        input_df = self.post_update(input_df)
        input_df = input_df.reset_index()
        self.df = input_df[[input_df.columns[0]]+orig_cols] if orig_cols else input_df
        self.init_x1 = self.xlen + defs.right_margin_candles - defs.side_margin
        self.cache_hilo = OrderedDict()
        self._period = self._smooth_time = None

    def set_df(self, df):
        self.df = df
        self.cache_hilo = OrderedDict()
        self._period = self._smooth_time = None

    def hilo(self, x0, x1):
        '''Return five values in time range: t0, t1, highest, lowest, number of rows.'''
        if x0 == x1:
            x0 = x1 = int(x1)
        else:
            x0,x1 = int(x0+0.5),int(x1)
        query = '%i,%i' % (x0,x1)
        if query not in self.cache_hilo:
            v = self.cache_hilo[query] = self._hilo(x0, x1)
        else:
            # re-insert to raise prio
            v = self.cache_hilo[query] = self.cache_hilo.pop(query)
        if len(self.cache_hilo) > 100: # drop if too many
            del self.cache_hilo[next(iter(self.cache_hilo))]
        return v

    def _hilo(self, x0, x1):
        df = self.df.loc[x0:x1, :]
        if not len(df):
            return 0,0,0,0,0
        timecol = df.columns[0]
        t0 = df[timecol].iloc[0]
        t1 = df[timecol].iloc[-1]
        valcols = df.columns[self.scale_cols]
        hi = df[valcols].max().max()
        lo = df[valcols].min().min()
        return t0,t1,hi,lo,len(df)

    def rows(self, colcnt, x0, x1, yscale, lod=True):
        df = self.df.loc[x0:x1, :]
        if self.is_sparse:
            df = df.loc[df.iloc[:,self.col_data_offset].notna(), :]
        origlen = len(df)
        return self._rows(df, colcnt, yscale=yscale, lod=lod), origlen

    def _rows(self, df, colcnt, yscale, lod):
        if lod and len(df) > defs.lod_candles:
            df = df.iloc[::len(df)//defs.lod_candles]
        colcnt -= 1 # time is always implied
        colidxs = [0] + list(range(self.col_data_offset, self.col_data_offset+colcnt))
        dfr = df.iloc[:,colidxs]
        if yscale.scaletype == 'log' or yscale.scalef != 1:
            dfr = dfr.copy()
            for i in range(1, colcnt+1):
                if dfr.iloc[:,i].dtype != object:
                    dfr.iloc[:,i] = yscale.invxform(dfr.iloc[:,i])
        return dfr

    def __eq__(self, other):
        return id(self) == id(other) or id(self.df) == id(other.df)

    def __hash__(self):
        return id(self)


