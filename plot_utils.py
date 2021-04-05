import numpy as np
import plotly.graph_objs as go
import plotly.express as px

import matplotlib.pyplot as plt
# https://pypi.org/project/hcp-utils/
import hcp_utils as hcp # hcp mesh, bgmap, etc
import nilearn.plotting as plotting
from nilearn.plotting import view_surf
from nilearn.plotting import plot_surf_stat_map as ps
from nilearn.plotting import img_plotting as ip
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable, get_cmap

def _hex_to_rgb(hex):
    '''
    for opacity of shaded error bars
    '''
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i+hlen//3], 16) 
        for i in range(0, hlen, hlen//3))

def _get_ts(a, dim, mult=1):
    '''
    a: (k_time x k_seq) or (k_seq x k_time)
    dim: which(k_seq)
    mult: multiplier for ste

    return:
    ts: dict {'mean': (1 x time),
        'ste: (1 x time)}
    '''
    k_seq = a.shape[dim]
    print(f"k_seq{k_seq}")
    ts = {'mean': np.mean(a, axis=dim),
          'ste': mult/np.sqrt(k_seq) * np.std(a, axis=dim),
          'std': np.std(a, axis=dim)}
    
    return ts

def _plot_ts(ts, level, color, showlegend=False, name='', opacity=0.3, width=2, y_lim = [1.0, 0.0]):
    '''
    plot shaded error bars
    input:
        ts: dict {'mean': (1 x time),
            'ste: (1 x time)}
        color: hex
        name: legendname
    '''
    def rescale(x, l, h):
#         x -= l
#         x /= (h-l)
        return x
    
    
    # normalize to 0-1 scale
    low_raw = rescale(ts['mean'] - ts['ste'], y_lim[1], y_lim[0])
    high_raw = rescale(ts['mean'] + ts['ste'], y_lim[1], y_lim[0])
    y_raw = rescale(ts['mean'], y_lim[1], y_lim[0])
    
    
    low = level + low_raw #ts['mean'] - ts['ste']
    high = level + high_raw #ts['mean'] + ts['ste']
    y = level + y_raw #ts['mean']
    k_time = len(ts['mean'])
    x = [(ii + 1) for ii in range(k_time)]
    
    fillcolor = _hex_to_rgb(color) + (opacity,) #opacity
    fillcolor = 'rgba' + str(fillcolor)
    
    # lowerbound
    lb = go.Scatter(name=name,
        x=x, y=low,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        showlegend=False,
        legendgroup=name)
    
    # plot line
    tr = go.Scatter(name=name,
        x=x, y=y,
        mode='lines',
        line=dict(color=color,
            width=width),
        fillcolor=fillcolor,
        fill='tonexty',
        showlegend=showlegend,
        legendgroup=name)
    
    # upperbound
    ub = go.Scatter(name=name,
        x=x, y=high,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor=fillcolor,
        fill='tonexty',
        showlegend=False,
        legendgroup=name)

    plotter = {'lb': lb, 'tr': tr, 'ub': ub}

    return plotter

def _plot_symmetric_ts(ts, level, color, 
    showlegend=False, name='', width=2):
    '''
    plot symmetric ts
    input:
        ts: dict {'mean': (1 x time),
            'ste: (1 x time)}
        color: hex
        name: legendname
    '''
    low = level - ts
    high = level + ts
    k_time = len(ts)
    x = [(ii + 1) for ii in range(k_time)]
    
    fillcolor = _hex_to_rgb(color) + (0.3,) #opacity
    fillcolor = 'rgba' + str(fillcolor)
    
    # lowerbound
    lb = go.Scatter(name=name,
        x=x, y=low,
        line=dict(color=color,
            width=width),
        mode='lines',
        showlegend=showlegend,
        legendgroup=name)
    
    # upperbound
    ub = go.Scatter(name=name,
        x=x, y=high,
        mode='lines',
        line=dict(color=color,
            width=width),
        fillcolor=fillcolor,
        fill='tonexty',
        showlegend=False,
        legendgroup=name)

    plotter = {'lb': lb, 'ub': ub}

    return plotter

def _add_box(i, j):
    '''
    box around heatmap entry
    for highlighting
    '''
    box = {}
    line = dict(color='#00cc96', width=4)

    box['l'] = go.Scatter(x=[j-0.5, j-0.5],
        y=[i-0.5, i+0.5],
        mode='lines',
        line=line,
        showlegend=False)
    box['r'] = go.Scatter(x=[j+0.5, j+0.5],
        y=[i-0.5, i+0.5],
        mode='lines',
        line=line,
        showlegend=False)
    box['u'] =  go.Scatter(x=[j-0.5, j+0.5],
        y=[i+0.5, i+0.5],
        mode='lines',
        line=line,
        showlegend=False)
    box['d'] = go.Scatter(x=[j-0.5, j+0.5],
        y=[i-0.5, i-0.5],
        mode='lines',
        line=line,
        showlegend=False)
    
    return box

def _highlight_max(fig, z, axis=0):
    '''
    helper to highlight max in each row/col
    '''
    kk = z.shape[axis]
    for ii in range(kk):
        jj = np.argmax(z[ii])
        box = _add_box(ii, jj)
        for line in box:
            fig.add_trace(box[line])
    
    return fig

def _trajectory3D(ts, color, 
    name='', lg='', showlegend=False,
    start=None, stop=None):
    '''
    plot trajectories in 3D
    used for mean trajectories
    input:
        ts: (3 x k_time)
        color: hex or rgb
        name: legendname
        lg: legend group

        start, stop: plotting interval
    '''
    
    if start == None:
        # handles rest period plotting
        start = 0
        stop = ts.shape[0]
        
    sc = go.Scatter3d(x=ts[start:stop, 0],
        y=ts[start:stop, 1],
        z=ts[start:stop, 2],
        marker=dict(size=5, color=color),
        line=dict(width=10, color=color),
        mode='lines+markers',
        name=name,
        showlegend=showlegend,
        legendgroup=lg)
    
    return sc

def _variance3D(ts, color, 
    name='', lg='', showlegend=False,
    start=None, stop=None):
    '''
    visualize variance across 3D trajectory
    input:
        ts: (3 x k_time)
        color: hex or rgb
        name: legendname
        lg: legend group

        start, stop: plotting interval
    '''
    if start == None:
        start = 0
        stop = ts.shape[0]
        
    sc = go.Scatter3d(x=ts[start:stop, 0],
        y=ts[start:stop, 1],
        z=ts[start:stop, 2],
        marker=dict(size=2.4, color=color),
        opacity=0.25,
        mode='markers',
        name=name,
        showlegend=showlegend,
        legendgroup=lg)

    return sc

def _big_markers(ts, color, start, lg=''):
    '''
    big markerks to indicate start of trajectory
    input:
        ts: (3 x k_time)
        color: hex or rgb
        lg: legend group

        start: start of trajectory
    '''

    sc = go.Scatter3d(x=[ts[start, 0]],
        y=[ts[start, 1]],
        z=[ts[start, 2]],
        marker=dict(size=10, color=color),
        mode='markers',
        showlegend=False,
        legendgroup=lg)
    
    return sc

def _get_norm(ts):
    '''
    input:
    ts: k_time x plot_dim
    output:
    compute norm at each time point
    dist_ts: k_time x 1 
    '''
    k_time = ts.shape[0]
    dist_ts = np.zeros(k_time)
    for ii in range(k_time):
        dist_ts[ii] = np.linalg.norm(ts[ii, :])
        
    return dist_ts
    
def _plot_hcp_style(x, title=None, figsize=None, vmax=None, 
    cmap='YlGnBu', display=False, colorbar=False):
    '''
    x: statmap
    title: fig title
    figsize = figsize
    customized based on existing nilearn functions
    some lines may not make sense :(
    '''
    
    if not figsize:
        figsize = (3, 2.5)
    
    if not vmax:
        vmax = np.max(x)
    
    if np.sum([x>0]) > 0:
        threshold = np.min(x[x>0]) # set threshold to eliminate artifacts
    else:
        threshold=0.005

    # the 4 views
    mesh = [hcp.mesh.inflated_right, hcp.mesh.inflated_left, 
            hcp.mesh.inflated_left, hcp.mesh.inflated_right]
    stat_map = [hcp.right_cortex_data(x), hcp.left_cortex_data(x),
                hcp.left_cortex_data(x), hcp.right_cortex_data(x)]
    bg_map = [hcp.mesh.sulc_right, hcp.mesh.sulc_left,
              hcp.mesh.sulc_left, hcp.mesh.sulc_right]
    view = ['medial', 'lateral', 'medial', 'lateral']
    
    # plot
    fig, axes = plt.subplots(nrows=2, ncols=2, 
                         subplot_kw={'projection': '3d'})
    fig.set_size_inches(figsize)
    for ii, ax in enumerate(axes.flatten()):
        ps(mesh[ii], stat_map[ii], 
           bg_map=bg_map[ii], threshold=threshold, 
           cmap=cmap,
           colorbar=False, 
           bg_on_data=True, darkness=0.2,
           view=view[ii], vmax=vmax, axes=ax)
    if title:
        fig.suptitle(title, fontsize=11, fontname="Nimbus")
    fig.tight_layout() # move subplots closer
    
    if colorbar:
        fig1, ax = plt.subplots()
        sm = _add_colorbar(x, vmax=1, cmap=cmap)
        # fake up the array of the scalar mappable.
        sm._A = []
        cbar_ax = fig1.add_axes([0, 0, 0.03, 1]) #fig.add_subplot(1, 32, 31)
        ax.remove()
        fig1.colorbar(sm, cax=cbar_ax, orientation='vertical')
        
    
    fig.subplots_adjust(wspace=-0.25, hspace=-0.40, top=1, bottom=0)
    if not display:
        plt.close()
    
    if colorbar:
        return fig, fig1
    return fig

def _add_colorbar(x, vmax, cmap):
    '''
    add colorbar
    helper for _plot_hcp_style
    '''
    cbar_vmin, cbar_vmax, vmin, vmax = ip._get_colorbar_and_data_ranges(
        x, vmax, symmetric_cbar='auto', kwargs='')
    print(np.min(x), np.max(x))
    # overwrite variables
    vmin, threshold = 0, 0
    
    #cmap = get_cmap('hot')
    cmap = get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # set colors to grey for absolute values < threshold
    istart = int(norm(-threshold, clip=True) * (cmap.N - 1))
    istop = int(norm(threshold, clip=True) * (cmap.N - 1))
    for i in range(istart, istop):
        cmaplist[i] = (0.5, 0.5, 0.5, 1.)
    our_cmap = LinearSegmentedColormap.from_list('Custom cmap',
                                                 cmaplist, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=our_cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    
    return sm