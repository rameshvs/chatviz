from __future__ import division

import random
import sys
import os.path
import tempfile
import cPickle

import matplotlib.pyplot as plt

import chatviz as cv
import chatviz.server

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('--plots', '--server'):
        print('USAGE: [%s --plots] or [%s --server]' % (sys.argv[0], sys.argv[0]))
        sys.exit(1)
    source_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.join(source_dir,'examples')

    (date_lists, counter_lists, words, intervals) = \
            cv.logread.generate_everything([('aim', ('romeo',), ('juliet',))],
                                           example_dir,
                                           None)

    if sys.argv[1] == '--plots':
        cv.plots.plot_all(date_lists, counter_lists, ('Romeo', 'Juliet'))
        plt.show()
    elif sys.argv[1] == '--server':
        f = tempfile.NamedTemporaryFile(delete=False)
        cPickle.dump(date_lists, f)
        cPickle.dump(counter_lists, f)
        f.close()
        port = random.randint(40000, 60000)
        print("*********************************************")
        print("******  Starting server on port %05d  ******" % port)
        print("******     Press Ctrl-C to quit...     ******")
        print("*********************************************")
        cv.server.main(dump_filename=f.name, port=port, bin_size=2)

        os.unlink(f.name)

