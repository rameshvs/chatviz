"""
A module for plotting the results of parsing instant message logs.

Author: Ramesh Sridharan (ramesh.eecs <at> gmail.com)
"""
from __future__ import division
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from chatviz import util

DEBUG = True
try:
    from IPython.core.debugger import Tracer
    if DEBUG:
        debug_here = Tracer()
    else:
        debug_here = lambda : None
except:
    debug_here = lambda : None

# TODO questions that might be interesting to answer:
# - how do words vary across time of day? across people?
#   - classification/feature importance?
# - how do my words vary across conversations with others?
# - are certain words/phrases correlated with high/low activity?
# - clusters of words!! LDA

ONE_DAY = datetime.timedelta(days=1)
TWO_WEEKS = datetime.timedelta(weeks=2)
MIDNIGHT = datetime.datetime(2004,1,1,0,0,0)
SECONDS_IN_A_DAY = 24 * 60 * 60
ALL_STYLES = ['r','b','g']

VERBOSE_MODE = True

def plot_all(datetime_lists, counter_lists, labels):
    """ Convenience function: plots results from two people """
    colors = ['#ff1133', '#3311ff']
    #colors = ['#00ff00', '#ffffff']
    plt.figure()
    top_ax = plt.subplot(2,1,1)
    counts = [[ctr.total() for ctr in counters] for counters in counter_lists]
    plot_cumulative_volume(datetime_lists, counts, ax=top_ax, labels=labels, styles=colors)

    bottom_ax = plt.subplot(2,1,2)
    grids = []
    for i in (0,1):
        grids.append(plot_conversation_density(datetime_lists[i],
                                               counts[i],
                                               bottom_ax,
                                               color=colors[i]))

    # Now, show conversation frequency
    plt.figure()
    ax = plt.gca()
    nonzero_counts = [grid[grid > 0] for grid in grids]
    maximum = max(map(np.max, nonzero_counts))
    ax.hist(nonzero_counts, bins=np.arange(maximum+1), label=labels, color=colors)
    plt.legend()



def plot_conversation_density(datetimes, counts, ax=None,
                              alpha=0.04, scale=2000, color='b'):
    """
    Plots conversation density by time of day as a scatterplot.

    Inputs
    ------
    datetimes : a (single) list of datetimes
    counts    : a (single) list of word counts for each datetime

    ax        : a matplotlib axis to plot into
    alpha     : the transparency of each scatter circle
    scale     : the data in counts are rescaled to be between 0 and this 
                for plotting
    color     : color for the scatterplot

    Returns the data in the grid as a 2D array (date x hour)

    For example:
        (date_lists, counter_lists) = chatviz.logread.generate_everything(...)

        dates = reduce(lambda x,y: x+y, date_lists)
        dates = [date for date_list in date_lists for date in date_list]

        counts = [ctr.total() for ctr_list in counter_lists for ctr in ctr_list]
        chatviz.plots.plot_conversation_density(dates,counts)
    """
    # TODO: implement this in JS, where mouseover shows the top words
    N = len(counts)
    assert len(counts) == len(datetimes), "Inputs should be the same length"
    dates = np.zeros(N)
    times = np.zeros(N)
    counts = np.array(counts)
    for (i, datetime) in enumerate(datetimes):
        dates[i] = datetime.toordinal()
        times[i] = 2+(util.seconds_since_midnight(datetime.time()) / SECONDS_IN_A_DAY)

    (min_date,max_date) = (min(dates), max(dates))
    (min_time,max_time) = (min(times), max(times))

    date_edges = np.arange(min_date, max_date+1)
    time_edges = np.linspace(2,3, len(date_edges))

    # use histogram2d for quantization: the histogram is like a grid
    (grid, _, _) = np.histogram2d(dates,
                                  times,
                                  bins=(date_edges, time_edges),
                                  weights=counts)
    bin_centers = []
    for edge_array in (date_edges, time_edges):
        bin_centers.append(.5*(edge_array[1:] + edge_array[:-1]))
    (date_centers, time_centers) = bin_centers
    nx = len(date_centers)
    ny = len(time_centers)

    # loop over each grid point, keeping its x and y values
    scatter_points = []
    for x in xrange(nx): # x represents date
        for y in xrange(ny): # y represents time of day
            value = grid[x,y]
            if value > 0:
                scatter_points.append((date_centers[x],time_centers[y],value))

    # use those grid points to create a scatterplot
    (xs, ys, cs) = zip(*scatter_points)
    cs = np.array(cs)
    cs = cs / np.max(cs) * scale
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.scatter(xs, ys, cs, alpha=alpha, edgecolors='none', c=color)

    ax.axis([date_edges[0],date_edges[-1],time_edges[-1],time_edges[0]])

    ## plot formatting
    ax.set_xlabel('Date')
    ax.xaxis_date()
    tick_labels = ax.get_xticklabels()
    for label in tick_labels:
        label.set(rotation=10)


    ax.set_ylabel('Time of day')
    yaxis_formatter = mdates.DateFormatter('%H:%M')
    ax.yaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.yaxis.set_major_formatter(yaxis_formatter)
    ax.yaxis_date()

    ax.set_title('Conversation density')

    # TODO returning this here is a little unclean: maybe refactor grid
    # computation code into a separate function?
    return grid


def plot_cumulative_volume(date_lists, count_lists,
                           ax=None, labels=None, styles=ALL_STYLES):
    """
    Plots cumulative conversation volume.

    Inputs
    ------
    date_lists  : a list of lists of datetimes. 1 list of dates per person.
    count_lists : a list of lists of counts. 1 list of counts per person.

    ax -- a matplotlib axis to plot into
    labels -- a list with names for each person.

    date_lists, counter_lists, and labels should all have the same length.

    Returns nothing

    For example:
        (date_lists, counter_lists) = chatviz.logread.generate_everything(...)
        cl = map(lambda counters: map(counter2counts, counters),
                 counter_lists)
        chatviz.plots.plot_cumulative_volume(date_lists,counter_lists)
    """
    if ax is None:
        ax = plt.figure().gca()

    if labels is None:
        labels = [None] * len(date_lists)
        show_legend = False
    else:
        show_legend = True

    for (dates, counters, style, label) in \
            zip(date_lists, count_lists, styles, labels):
        ax.plot_date(dates, np.cumsum( counters ), style, label=label, linewidth=2)
        print("Plotted!")

    ## plot formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of words sent')
    ax.set_title('Cumulative wordcounts')
    myax = list(ax.axis('tight'))
    xaxis_padding = 10 # this is in days. TODO use date abstraction barrier?

    myax[0] -= xaxis_padding
    myax[1] += xaxis_padding
    ax.axis(myax)

    if show_legend:
        ax.legend(loc=2)

    tick_labels = ax.get_xticklabels()
    for label in tick_labels:
        label.set(rotation=10)

