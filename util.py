"""
A module with misc. utility functions for IM log visualization

Author: Ramesh Sridharan (ramesh.eecs <at> gmail.com)
"""
import collections
from os import path
import string
import itertools

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

## Stop words are stored in the repository at the same location as this file:
stopword_file = path.join(path.dirname(path.abspath(__file__)),
                          'stopwords.txt')

with open(stopword_file, 'r') as f:
    STOP_WORDS = f.read().split()

def counter2counts(counter):
    return sum(counter.values())

def seconds_since_midnight(time):
    return time.second + (60 * (time.minute + 60 * time.hour))

def renormalize(data,(newmin,newmax),oldrange=None):
    """
    Linearly rescales data to lie between newmin and
    newmax. oldrange defaults to the min/max of data.
    This is similar to MATLAB's mat2gray, but with no clipping.
    """
    data = data.astype('float64')
    if oldrange is None:
        (oldmin,oldmax) = (np.min(data),np.max(data))
    else:
        (oldmin,oldmax) = oldrange
    slope = (newmin-newmax+0.)/(oldmin-oldmax)
    out = slope*(data-oldmin) + newmin
    return out

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def flatten_itertools(list_of_lists):
    return itertools.chain.from_iterable(list_of_lists)

def simplify_SN(SN):
    return SN.lower().replace(' ','')

def crossed_day(before, after):
    # Could do something more sophisticated, but this will do for now.
    return (after.hour == 0 and before.hour == 23)

def make_ngram_counter(N,instring,ignore_case=True,
                       should_strip_punctuation=True):
    if ignore_case:
        instring = instring.lower()
    if should_strip_punctuation:
        instring = strip_punctuation(instring)
    words = instring.split()
    n = len(words) - (N-1)
    ng = [' '.join(words[k:k+N]) for k in xrange(n)]

    wc = WordCounter()
    for ngram in ng:
        wc[ngram] += 1
    return wc

def strip_punctuation(mystring):
    return mystring.translate(string.maketrans('',''),string.punctuation)

def list_of_children(list_of_parents):
    return reduce(lambda x,y: x+y,
                  map(lambda x: x.getchildren(),
                      list_of_parents))

class WordCounter(collections.Counter):
    def without_below_threshold(self,threshold):
        for word in self.keys():
            if self[word] <= threshold:
                self.pop(word)

    def remove_below_threshold(self, threshold):
        new = self.copy()
        new.without_below_threshold(threshold)
        return new

    def remove_stopwords(self,stopwords=STOP_WORDS):
        for word in self.keys():
            if word in stopwords:
                self.pop(word)

    def without_stopwords(self, stopwords=STOP_WORDS):
        new = self.copy()
        new.remove_stopwords(stopwords)
        return new

    def total(self):
        return sum(self.itervalues())

    def top_N(self,N):
        top = self.most_common(N)
        return sorted(top)

    # collection.Counter's versions of these are not actually
    # in place (and are therefore too slow)
    def __iadd__(self,other):

        for word in other:
            self[word] += other[word]
        return self

    def __isub__(self,other):

        for word in other:
            self[word] -= other[word]
            assert self[word] >= 0
            if self[word] == 0:
                self.pop(word)
        return self

