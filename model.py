"""
A simple model for conversations. Similar to a first-order Markov
model, but every word has a probability of being independent that's
related to how long it was since the previous message.

Let w_i and t_i be the word at time i and the gap (in seconds)
between the sending of word i-1 and word i. Let c_i be 1 if word i
depends on word i-1, and 0 otherwise.
p(c_i) = exp(-t_i)
If c_i = 1, w_i is drawn from some transition distribution at w_{i-1}.
If c_i = 0, w_i is drawn from some refresh distribution.

Author: Ramesh Sridharan (ramesh.eecs <at> gmail.com)
"""
from __future__ import division

import numpy as np
import scipy.sparse as sparse

def intervals_to_probabilities(intervals, lambd):
    """ Probability of staying on topic from word to word p(c_i) """
    # TODO maybe have lower bound above 0 (i.e. exp(...) + .001)?
    intervals_arr = np.array(intervals)
    probs = np.exp(-intervals_arr * lambd)
    return probs

def normalize_rows(csr_matrix):
    """ Makes rows sum to 1 for use as a (row-stochastic) distribution """
    # TODO investigate csr_scale_rows again?
    row_sums = np.array(csr_matrix.sum(1).astype('float64')).squeeze()
    m = csr_matrix.copy().astype('float64')
    m.eliminate_zeros()
    (row_indices, col_indices) = csr_matrix.nonzero()
    m.data /= row_sums[row_indices]
    return m

def sample(pmf):
    """ Samples from a PMF """
    if sparse.issparse(pmf):
        (rows, cols) = pmf.nonzero()
        pmf = np.array(pmf[rows,cols])[0,:]
        was_sparse = True
    else:
        was_sparse = False
    sample = np.argmax(np.random.multinomial(1, pmf))
    if was_sparse:
        sample = cols[sample]
    assert sample != 0

    return sample

class TransitionModel(object):
    """
    Encapsulates relevant bits for an instantiation of the model described
    in the module. Each word is mapped to a unique whole number index to make
    array indexing easier.

    Key instance variables:
    -----------------------
    all_words : Maps index to word (i.e. list of words in indexing order)
    mapping   : Maps word to index (dict)
    transition_distribution : entry (i,j) represents the probability of
                              going from word i to word j (used when c_i=1)
    marginal_distribution   : entry i represents the probability of
                              ``resetting'' with word i (used when c_i = 0)

    Useful Methods
    -------
    train()                           : learns parameters from given words
    generate_sample(N)                : generates a sample of N words
    get_transition_distribution(word) : gets distribution over the next word
    """

    def __init__(self, words, intervals, threshold=.02, lambd=.0173):
        """
        Inputs
        ------
        words : a (flat) list of words from the conversation (list of w_i)
        intervals : a list of time intervals between words (list of t_i)
        threshold : Convergence threshold for EM while training
        lambd : Parameter for whether to reset (see intervals_to_probabilities)
                lower -> less likely to reset, higher -> more likely to reset
        """
        assert len(words) == len(intervals) + 1

        self.intervals = intervals

        self.threshold = threshold
        self.lambd = lambd

        self.all_words = sorted(set(words))
        self.mapping = dict((wd, i) for (i,wd) in enumerate(self.all_words))
        self.indices = map(lambda wd: self.mapping[wd], words)

    def _sample_word(self, prev=None):
        """ Samples one word given previous word (internal) """
        if prev is None:
            pmf = self.marginal_distribution
        else:
            pmf = self.transition_distribution[prev,:]
            if pmf.nnz == 0:
                pmf = self.marginal_distribution
        idx = sample(pmf)
        return idx

    def draw_sample(self, N, interval_pmf=None, length_pmf=None):
        """
        Draws a sample from the learned transition distribution.

        Inputs
        ------
        interval_pmf : distribution over interval lengths between messages
        length_pmf   : distribution over message lengths

        Returns (indices, words, intervals)
        -------
        indices : list of lists of sampled indices (one list per message)
        words   : the corresponding sampled words
        intervals : the sampled intervals between messages
        """
        # TODO add message style formatting,
        if interval_pmf is None and length_pmf is None:
            interval_pmf = 1/np.arange(2000)
            interval_pmf /= interval_pmf.sum()

        if length_pmf is None:
            length_pmf = 1 / np.arange(40)
            length_pmf /= length_pmf.sum()

        indices = [[self._sample_word()]]
        intervals = []
        while len(indices) < N:
            message_length = 0
            while message_length == 0:
                message_length = sample(length_pmf)
            message = []
            message_intervals = [sample(interval_pmf)]
            message_intervals.extend([0] * (message_length-1))
            ontrack_prob = np.exp(message_intervals[0] * self.lambd)
            if ontrack_prob > np.random.random_sample():
                prev = None
            else:
                prev = indices[-1][-1]
            for i in xrange(message_length):
                idx = self._sample_word(prev)
                assert idx != 0
                message.append(idx)
                prev = idx
            indices.append(message)
            intervals.append(message_intervals)
        words = [[self.all_words[idx] for idx in msg] for msg in indices]
        return (words, indices, intervals)

    def get_transition_distribution(self, word):
        """ Given a word, gives the distribution over possible next words """
        assert self.convergence_reached , "Need to run train() first"
        idx = self.mapping[word]
        row = self.transition_distribution[idx,:]
        (_, columns) = row.nonzero()
        distribution = {}
        for c in columns:
            distribution[self.all_words[c]] = row[0,c]
        return distribution

    def train(self):
        """
        Uses EM to learn parameters. Computes soft estimates for c_i
        (ontrack_probs), and uses those to learn parameters
        (transition_distribution and marginal distribution).
        See also _learn_parameters() (M step) and _infer_ontrack_probs (E step)
        """
        self.priors = intervals_to_probabilities(self.intervals,
                                                      self.lambd)
        self.ontrack_probs = self.old_ontrack_probs = self.priors
        self.convergence_reached = False
        self.n_iters = 0
        while not self.convergence_reached:

            self._learn_parameters()    # M-step
            self._infer_ontrack_probs() # E-step

            ### Check for convergence
            meandiff = np.mean(np.abs(self.old_ontrack_probs - \
                                        self.ontrack_probs))
            if meandiff < self.threshold:
                self.convergence_reached = True
            print(meandiff)

            self.old_ontrack_probs = self.ontrack_probs
            self.n_iters += 1

    def _learn_parameters(self):
        """ M-step: estimates parameters from data & ontrack_probs """
        soft_counts = sparse.coo_matrix( (self.ontrack_probs,
                                          (self.indices[:-1],
                                           self.indices[1:]  )) )
        self.soft_counts = soft_counts.tocsr()
        self.marginal_distribution = np.bincount(self.indices[1:], 
                                                 1-self.ontrack_probs)
        self.marginal_distribution /= np.sum(self.marginal_distribution)

        self.transition_distribution = normalize_rows(self.soft_counts)

        self.transition_probs = self.transition_distribution[self.indices[:-1],
                                                             self.indices[1:]]

    def _infer_ontrack_probs(self):
        """ E-step: estimates ontrack_probs from data & parameters """
        self.weight_ontrack = self.priors * \
                              np.array(self.transition_probs).squeeze()
        self.weight_switch = (1-self.priors) * \
                        self.marginal_distribution[self.indices[1:]]

        self.ontrack_probs = (self.weight_ontrack) / \
                             (self.weight_ontrack + self.weight_switch)
