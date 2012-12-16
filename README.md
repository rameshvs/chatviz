# chatviz
## A library for visualizing your IM logs

This is a python library for visualizing your IM logs. Here are some
examples of what it can do:

*   Plot how much stuff you say using IMs (top) and how your conversation volume
    changes by time of day (bottom):

    ![Conversation volume plot and density scatterplot](https://github.com/rameshvs/chatviz/raw/master/screenshots/plots.png)

*   Visualize your conversation topics over time with a dynamic wordcloud

    ![Dynamic wordcloud 1](https://github.com/rameshvs/chatviz/raw/master/screenshots/wordcloud1.png)
    ![Dynamic wordcloud 2](https://github.com/rameshvs/chatviz/raw/master/screenshots/wordcloud2.png)

You'll need NumPy for just about everything,
[matplotlib](http://matplotlib.org/ 'matplotlib') for generating plots with
`plots.py`, and [CherryPy](http://www.cherrypy.org/ 'CherryPy') for running
the word cloud server in `server.py`.

## Getting Started

You can use the sample data in `examples/`, which contains all dialog from
_Romeo and Juliet_ by the two protagonists in [Pidgin](http://pidgin.im/
'pidgin') IM format. `python starter.py --plots` or `python starter.py --server`
will show the matplotlib plots or start the word cloud server on `localhost`
respectively.

You can use the functions in `logread.py` with your own pidgin logs, which are
usually in `$HOME/.purple/logs`.

