import collections
from os import path

## Stop words are stored in the repository at the same location as this file:
stopword_file = path.join(path.dirname(path.abspath(__file__)),
                          'stopwords.txt')
with open(stopword_file, 'r') as f:
    STOP_WORDS = f.read().split()

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

class WordCounter(collections.Counter):
    def remove_below_threshold(self,threshold):
        new = self.copy()
        for word in self:
            if self[word] <= threshold:
                new.pop(word)
        return new

    def remove_stopwords(self,stopwords=STOP_WORDS):
        new = self.copy()
        for word in self:
            if word in stopwords:
                new.pop(word)
        return new

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

