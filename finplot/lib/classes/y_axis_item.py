import pyqtgraph as pg

class YAxisItem(pg.AxisItem):
    def __init__(self, vb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vb = vb
        self.hide_strings = False
        self.style['autoExpandTextSpace'] = False
        self.style['autoReduceTextSpace'] = False
        self.next_fmt = '%g'

    def tickValues(self, minVal, maxVal, size):
        vs = super().tickValues(minVal, maxVal, size)
        if len(vs) < 3:
            return vs
        xform = self.vb.yscale.xform
        gs = ['%g'%xform(v) for v in vs[2][1]]
        maxdec = max([len((g).partition('.')[2].partition('e')[0]) for g in gs])
        if any(['e' in g for g in gs]):
            self.next_fmt = '%%.%ig' % maxdec
        else:
            self.next_fmt = '%%.%if' % maxdec
        return vs

    def tickStrings(self, values, scale, spacing):
        if self.hide_strings:
            return []
        xform = self.vb.yscale.xform
        return [self.next_fmt%xform(value) for value in values]
