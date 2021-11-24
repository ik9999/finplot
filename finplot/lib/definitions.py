from datetime import datetime, timezone
import pyqtgraph as pg

# appropriate types
ColorMap = pg.ColorMap

# module definitions, mostly colors
legend_border_color = '#777'
legend_fill_color   = '#6668'
legend_text_color   = '#ddd6'
soft_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
hard_colors = ['#000000', '#772211', '#000066', '#555555', '#0022cc', '#ffcc00']
colmap_clash = ColorMap([0.0, 0.2, 0.6, 1.0], [[127, 127, 255, 51], [0, 0, 127, 51], [255, 51, 102, 51], [255, 178, 76, 51]])
foreground = '#000'
background = '#fff'
odd_plot_background = '#fff'
candle_bull_color = '#26a69a'
candle_bear_color = '#ef5350'
candle_bull_body_color = background
candle_shadow_width = 1
volume_bull_color = '#92d2cc'
volume_bear_color = '#f7a9a7'
volume_bull_body_color = volume_bull_color
volume_neutral_color = '#bbb'
poc_color = '#006'
band_color = '#d2dfe6'
cross_hair_color = '#0007'
draw_line_color = '#000'
draw_done_color = '#555'
significant_decimals = 8
significant_eps = 1e-8
max_zoom_points = 20 # number of visible candles when maximum zoomed in
top_graph_scale = 2
clamp_grid = True
right_margin_candles = 5 # whitespace at the right-hand side
side_margin = 0.5
lod_candles = 3000
lod_labels = 700
cache_candle_factor = 3 # factor extra candles rendered to buffer
y_pad = 0.03 # 3% padding at top and bottom of autozoom plots
y_label_width = 65
display_timezone = datetime.now(timezone.utc).astimezone().tzinfo  # default to local
winx, winy, winw, winh = 300,150,800,400
log_plot_offset = -2.2222222e-16 # I could file a bug report, probably in PyQt, but this is more fun
# format: mode, min-duration, pd-freq-fmt, tick-str-len
time_splits = [('years', 2*365*24*60*60,  'YS',  4), ('months', 3*30*24*60*60, 'MS', 10), ('weeks',   3*7*24*60*60, 'W-MON', 10),
               ('days',      3*24*60*60,   'D', 10), ('hours',        9*60*60, '3H', 16), ('hours',        3*60*60,     'H', 16),
               ('minutes',        45*60, '15T', 16), ('minutes',        15*60, '5T', 16), ('minutes',         3*60,     'T', 16),
               ('seconds',           45, '15S', 19), ('seconds',           15, '5S', 19), ('seconds',            3,     'S', 19),
               ('milliseconds',       0,   'L', 23)]

app = None
windows = [] # no gc
timers = [] # no gc
sounds = {} # no gc
epoch_period = 1e30
last_ax = None # always assume we want to plot in the last axis, unless explicitly specified
overlay_axs = [] # for keeping track of candlesticks in overlays
viewrestore = False
master_data = {}
