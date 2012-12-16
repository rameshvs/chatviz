"""
A module for a chat log visualization server.

Author: Ramesh Sridharan (ramesh.eecs <at> gmail.com)
"""
import datetime
import sys
import cPickle
import os
import string
import random
import json

try:
    import progressbar
except ImportError:
    progressbar = None

import numpy as np
import cherrypy

from chatviz import util

# TODO support more than 2 users at once
#  - pass in usernames/labels that can be used instead of _names_

# TODO save as json and avoid the need for a server altogether! this can all be
# done client-side, I think...

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTENT_DIR = os.path.join(CURRENT_DIR,'content/')
SPACE_CHARACTER = '_'
REMOVE_STOP_WORDS = True
LETTERS = string.ascii_uppercase + string.ascii_lowercase

def make_junk_words():
    # generate some junk words
    N_words = random.randrange(30,40)
    words = []
    counts = []
    for i in xrange(N_words):
        length = random.randint(4,8)
        count = 10+(random.random()*18) # 10 to 18
        word_list = (random.choice(LETTERS) for i in xrange(length))
        words.append(''.join(word_list))
        counts.append(count)
    return (words,counts)

def make_word_tag(identifier, word):
    formatter = '<a href="#" id="wordcloud%(who)s%(word_clean)s">%(word)s</a>'
    word_clean = word.replace(' ', SPACE_CHARACTER)
    return formatter % {'who':identifier, 'word':word, 'word_clean':word_clean}

def set_json():
    cherrypy.response.headers['Content-Type'] = 'application/json'

class WordCloudServer(object):
    """
    Server for word cloud visualization
    """

    def load_from_file(self, filename):
        """
        Loads data from filename. File must have:
          - list of 2 lists of dates and
          - list of 2 lists of counters
        (see logread.dump_output)
        """
        with open(filename,'r') as f:
            self.date_lists = cPickle.load(f)
            self.counter_lists = cPickle.load(f)
            assert f.read(1) == '', "File had more than I expected"
        assert len(self.date_lists) == len(self.counter_lists) == 2

    # TODO make font range/RGB tweakable on the website
    @cherrypy.expose
    def set_font_range(self, min_font_size, max_font_size):
        """ Sets minimum and maximum computed font sizes """
        self.font_range = (min_font_size, max_font_size)

    @ cherrypy.expose
    def set_RGBs(self, identifier, RGB_range): # numbers
        """ Sets color rangers to interpolate between for one person """
        self.RGB_ranges[identifier] = RGB_range

    def _set_number_of_words(self, N):
        """ Sets number of words displayed per person """
        self.N_words_to_display = N

    def __init__(self,data_filename, interval_size,
                 remove_stop_words=True, prune_rare_threshold=0):
        """
        Constructor.

        Inputs
        ------
        data_filename : a filename with pickled date lists and counter lists
        interval_size : size of the bins/intervals to graph words over

        remove_stop_words    : whether or not to remove stop words
        prune_rare_threshold : remove words below this threshold *before*
                               aggregating (for efficiency)
        """

        self.load_from_file(data_filename)

        if remove_stop_words:
            for person in (0,1):
                for counter in self.counter_lists[person]:
                    counter.remove_stopwords()
        print("Done removing stopwords. Binning words...")

        (ordinals, self.accumulator_lists) = intervalize_words(self.date_lists,
                                              self.counter_lists, interval_size)

        self.tdinterval = datetime.timedelta(interval_size)
        self.date_bins = map(datetime.datetime.fromordinal,ordinals)

        if prune_rare_threshold > 0:
            for accumulator_list in self.accumulator_lists:
                for counter in accumulator_list:
                    counter.remove_below_threshold(prune_rare_threshold)

        (self.start, self.end) = (0, len(self.date_bins) - 1)

        self.set_font_range(15,50)
        self.RGB_ranges = {}

        self.names = ('me', 'other') # all paired lists have same order as this
        self.set_RGBs('me', ((0xCC,0xBB,0xAA),(0x22,0x55,0xFF)))
        self.set_RGBs('other', ((0xAA,0xBB,0xCC),(0xFF,0x55,0x22)))
        self._set_number_of_words(20)

        print("Done binning words. Precomputing clouds...")
        self.cache = {}
        paired_counters = zip(*self.accumulator_lists)
        N = len(self.date_bins)
        # boundary is always 1 past where we can go
        direction = None
        end = 0
        N_entries = 0
        total_N_entries = (N * (N+1)) / 2
        if progressbar:
            pbar = progressbar.ProgressBar(maxval=total_N_entries).start()
        ## Algorithm:
        # Want to pre-compute sum over every possible interval.
        #  outer loop: "interval end" moves down
        #  inner loop: "interval start" moves up or down; it alternates
        #              over iterations of the outer loop. This way, it only
        #              moves by one each step, so we never have to reset the
        #              counter.
        # TODO optimize this loop a bit
        for start in xrange(N):
            if direction == 1:
                direction = -1
                boundary = start-1
                end = N-1
            elif direction == -1 or direction is None:
                interval_counters = [util.WordCounter() for persn in self.names]
                direction = 1
                end = start
                boundary = N
            else:
                raise ValueError("internal error -- invalid direction")
            #print("  Start position %d. Completed %d entries"%(start,N_entries))
            while end != boundary:
                final_json_dict = {}
                # loop over people
                for (interval,new,who) in zip(interval_counters,
                                            paired_counters[end],
                                            self.names):
                    if direction == 1:
                        interval += new
                    elif direction == -1:
                        interval -= new
                    else:
                        raise ValueError("internal error -- invalid direction")

                    top_words = interval.top_N(self.N_words_to_display)
                    final_json_dict[who] = self.to_dict_for_json(top_words,who)
                self.cache[(start,end)] = json.dumps(final_json_dict)
                N_entries += 1
                end += direction
            if progressbar:
                pbar.update(N_entries)
        if progressbar:
            pbar.finish()


    @cherrypy.expose
    def index(self):
        return open(os.path.join(CURRENT_DIR,u'index.html'))

    @cherrypy.expose
    def getsliderbins(self):
        set_json()
        out = {}
        for (who,accumulator_list) in zip(self.names, self.accumulator_lists):
            out[who] = [sum(c.values()) for c in accumulator_list]
        out['mindate'] = np.min(self.date_bins).isoformat()
        out['maxdate'] = np.max(self.date_bins).isoformat()
        return json.dumps(out)

    def to_dict_for_json(self,word_count_pairs,identifier):
        """
        Converts a list of (word,count) pairs into a json-style dict
        with the appropriate HTML for a word cloud.

        word_count_pairs is a list/tuple: ((word1,count1), (word2,count2),...)

        identifier is a member of self.names specifiying who the pairs are from
        """
        try:
            words,counts = zip(*word_count_pairs)
        except ValueError:
            json = {'textbody':''}
            return json

        count_strings = ['%0.3f'%x for x in util.renormalize(np.log(counts),
                                                            self.font_range)]
        RGB_range = zip(*self.RGB_ranges[identifier])
        RGB_values = [] # for each of R/G/B, an array of intensities per word
        for color in [0,1,2]: # r,g,b
            color_range = RGB_range[color]
            counts_as_colors = util.renormalize(np.log(counts),color_range)
            RGB_values.append(counts_as_colors.astype('int64'))
        RGB_strings = zip(*map(lambda colors: map(lambda color: '%02x'%color,
                                                   colors),
                                RGB_values))
        colors_hex = [r+g+b for (r,g,b) in RGB_strings]
        output = []
        for (count_string,rgb,count) in zip(count_strings,colors_hex,counts):
            output.append( count_string+','+rgb+','+str(count) )

        ids = ["wordcloud" + identifier + w.replace(' ',SPACE_CHARACTER) for w in words]

        json = dict(zip(ids,output))

        word_tags = (make_word_tag(identifier, w) for w in words)
        json['textbody'] = '\n'.join(word_tags)
        return json

    @cherrypy.expose
    def computewords(self):
        set_json()
        return self.cache[(self.start,self.end)]

    @cherrypy.expose
    def updatebounds(self,start,end):
        set_json()
        (self.start,self.end)= (int(start),int(end))

        datestrings = map(lambda idx: self.date_bins[idx].date().isoformat(),
                          (self.start,self.end))
        #datestrings = [for idx in (self.start,self.end)]
        return json.dumps(dict(zip(("start","end"),datestrings)))


def intervalize_words(date_lists, counter_lists, interval=14):
    """
    bins words in counters. takes in an interval (bin size in days), and any number
    of date-list/counter-list pairs. For example
    intervalize_words((dates1,counters1,dates2,counters2))

    returns one list of dates (bin boundaries), and several lists of counters that 
    accumulate values within bins (one list of counters for each input pair).
    """
    (minim, maxim) = padded_interval(np.hstack(date_lists), interval)
    rang = np.arange(minim.toordinal(), maxim.toordinal()+1, interval)
    all_accumulator_lists = []
    for (dates,counters) in zip(date_lists, counter_lists):
        accumulators = [util.WordCounter() for d in rang]
        indices = np.digitize([d.toordinal() for d in dates],rang)
        for (i,c) in zip(indices,counters):
            accumulators[i] += c
        all_accumulator_lists.append(accumulators)
    return (rang,all_accumulator_lists)

def padded_interval(dates,interval):
    dates = sorted(dates)
    (mini,maxi) = (dates[0],dates[-1])

    # have each one go until 5am (presumably nothing is happening then)
    mini -= datetime.timedelta(interval)
    mini = datetime.datetime(mini.year,mini.month,mini.day,5)

    maxi += datetime.timedelta(interval)
    maxi = datetime.datetime(maxi.year,maxi.month,maxi.day,5)

    return (mini,maxi)

def main(dump_filename, port, bin_size):
    global_config = { 'server.socket_port': port,
                      'server.socket_host': '0.0.0.0' }
    appconfig = { '/content': { 'tools.staticdir.on' : True,
                                'tools.staticdir.dir' : CONTENT_DIR }}
    cherrypy.config.update(global_config)
    cherrypy.quickstart(WordCloudServer(dump_filename,bin_size),
                        config=appconfig)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('USAGE: %s <logread dump> <port> <bin size in days>'%sys.argv[0])
        sys.exit(1)
    port = int(sys.argv[2])
    bin_size = int(sys.argv[3])
    dump_filename = sys.argv[1] # see logread.dump_output()
    main(dump_filename, port, bin_size)

