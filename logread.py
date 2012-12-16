"""
A module for parsing instant message logs.

Author: Ramesh Sridharan (ramesh.eecs <at> gmail.com)
"""
import os
import re
import cPickle
import itertools
import collections

import datetime
import dateutil.parser

import xml.etree.ElementTree as ET
from operator import itemgetter,methodcaller

from chatviz import util


#########################################
# Regular expressions for parsing lines #
#########################################

reflags = re.IGNORECASE

# TODO make more specific: this might match unwanted stuff (but it hasn't
# happened so far)
pidgin_session_start = re.compile(r'Conversation with .* at (.*) on .*',reflags)

# matches: 'Session Start (<me>:<them>): <timestamp>' with possible
# garbage characters at the start (not sure why they're there, but they are)
trillian_session_start = re.compile(r'[^\)^\]]*Session Start \([^\)]*\): (.*)',
                                    reflags)

# matches '(<timestamp>) <screenname>: blah blah' or
#         '[<timestamp>] <screenname>: blah blah'.
# (1) is the timestamp (complicated because we don't want to match extra
# stuff, and we want to allow it to be in brackets or parens). (2) is the 
# screenname sending the IM, and (3) is the message
#                    ***********(1)********* ****(2)***** *(3)*
my_im = re.compile(r'[\(\[]([^\)^\]]*)[\)\]] ([a-z0-9 ]*):(.*)',reflags)
# v-- this simpler version doesn't work because of pastes w/timestamps in them
#                    ******(1)******* ****(2)***** *(3)*
#my_im = re.compile(r'[\(\[](.*)[\)\]] ([a-z0-9 ]*):(.*)',reflags)


ONE_DAY = datetime.timedelta(days=1)
MIDNIGHT = datetime.datetime(2004,1,1,0,0,0)
STANDARD_TIMESTAMP = '%I:%M:%S %p'

#########################
# Convenience functions #
#########################
def generate_everything(sn_lists, logdir, trillian_logdirs,
                        pidgin_self_aliases=None, pidgin_buddy_aliases=None,
                        ngram_N=1, names=['me','other']):
    """
    Convenience function for aggregating across protocols/programs

    Inputs
    ------
    sn_lists : a list of things that look like 
               ('aim',['me1','me2',...],['other1','other2',...])
    logdir : where pidgin logs (if any) are stored
    trillian_logdirs : where trillian logs (if any) are stored
    pidgin_self_aliases : the result of calling PidginReader.make_my_aliases
    pidgin_buddy_aliases : the result of calling PidginReader.make_buddy_aliases
    ngram_N : 1 for unigrams, 2 for bigrams, etc.
    names : your name and the other person's name

    Returns
    -------
    date_lists  : a list of lists of datetimes. 1 list of dates per person.
    counter_lists : a list of lists of counters. 1 list of counters per person.
    words : 2 lists, each with all words used by that person
    intervals : 2 lists of intervals, giving the spacing between words (above)
    """
    import numpy as np

    pidgin_self_aliases = pidgin_self_aliases or \
            collections.defaultdict(lambda : [])
    pidgin_buddy_aliases = pidgin_buddy_aliases or \
            collections.defaultdict(lambda : [])
    readers = []
    for sn_list in sn_lists:
        (protocol,mes,others) = sn_list
        if protocol != 'trillian':
            assert protocol in ['aim', 'jabber', 'msn'] , "Unknown protocol %s"%protocol
            for (me, other) in itertools.product(mes,others):
                reader = PidginReader(logdir,
                                      [pidgin_self_aliases[me]+[me],
                                          pidgin_buddy_aliases[other]+[other]],
                                      [me,other],
                                      ngram_N,
                                      protocol)
                reader.read_all_files()
                readers.append(reader)
        else:
            for other in others:
                for trillian_logdir in trillian_logdirs:
                    reader = TrillianReader(trillian_logdir,
                    # I don't know how trillian generates its aliases...
                                            [mes, [util.simplify_SN(other)]],
                                            [None, other],
                                            ngram_N)
                    reader.read_all_files()
                    readers.append(reader)
    combined = LogReader.combined_results(True, *readers)

    keys = ['line_dates', 'line_counters', 'words', 'word_dates']
    (date_lists, counter_lists, words, word_date_lists) = \
        [[r[key] for r in combined] for key in keys]
    interval_tds = [np.diff(word_dates) for word_dates in word_date_lists]
    intervals = [map(methodcaller('total_seconds'),s) for s in interval_tds]

    return (date_lists, counter_lists, words, intervals)

def dump_output(outfile, date_lists, counter_lists):
    """ Pickles and dumps output to a file for use with server.py. """
    with open(outfile,'w') as f:
        cPickle.dump(date_lists, f)
        cPickle.dump(counter_lists, f)



#####################
# Log-reading logic #
#####################
class LogReader(object):
    """
    Class that reads IM log files from some location. Designed primarily
    with pidgin (and to a lesser extent trillian) in mind.
    """
    @classmethod
    def combined_results(cls, sort, *readers):
        # TODO rewrite this comment
        """
        input: N lists.
         => each of those lists has 2 sublists
          => each of those sublists has len(NRANGE)+1 items
        output: a list with 2 sublists
         => each sublist should have len(NRANGE)+1 items, and should
            be the result of combining the corresponding N items from input
        """
        separate_results = [reader.get_results() for reader in readers]
        combined_results = [{}, {}]
        for i in [0,1]:
            results = combined_results[i]
            for key in ['line_dates', 'line_counters', 'words', 'word_dates']:
                lists = [result[i][key] for result in separate_results]
                results[key] = reduce(lambda x,y: x+y, lists)
            for p in (('line_dates', 'line_counters'), ('word_dates', 'words')):
                srt = zip(*[results[key] for key in p])
                srt.sort(key=itemgetter(0))
                (results[p[0]], results[p[1]]) = zip(*srt)
        return combined_results


    def __init__(self, loghome, both_aliases, both_screennames, ngram_N):
        """
        both_aliases is a two-element list of lists.
        Convention for this list and others: first one is always self 
        (person storing logs), second is conversational partner
        """
        self.loghome = loghome
        self.both_aliases = both_aliases
        self.both_screennames = both_screennames
        self.ngram_N = ngram_N

        self.both_lines = [{}, {}]
        for i in (0,1):
            #self.both_lines[i] = {}
            for key in ['line_dates', 'line_counters', 'words', 'word_dates']:
                self.both_lines[i][key] = []
            #self.both_lines[i] = [ [] for k in xrange(1+1+2) ]
        # hopefully you don't have any logs dated before this ;)...
        self.previous = datetime.datetime(1,1,1)

    def read_file(self, filename):
        """ Reads a single log file """
        self.current_datetime = None
        with open(filename, 'r') as fp:
            # always check the first line for session info
            self.read_line(fp.readline(), True)
            for line in fp:
                self.read_line(line, self.always_check_session)

    def read_all_files(self):
        """
        Abstract method: should compute files from self.loghome and
        self.both_screennames and read them using read_file()
        Should only be called once: calling multiple times gives undefined
        behavior
        """
        raise NotImplementedError

    def get_results(self, sort=False):
        """
        For each person, returns a tuple of (timestamps, counters_1, ...)
        """
        if sort:
            for i in (0,1):
                print([len(x) for x in self.both_lines[i]])
                raise NotImplementedError
                self.both_lines[i] = zip(*sorted(zip(*self.both_lines[i])))
        return self.both_lines

    def read_line(self, line, allow_session_start=True):
        """
        Reads one line from an IM log. If it's the next IM in the sequence, adds
        the timestamp+words to the appropriate lists.
        """
        matcher = my_im.match(line)
        if matcher is None:
            if allow_session_start:
                ##### Check for session start
                session_start_match = self.session_start_matcher.match(line)
                if session_start_match:
                    self.previous = dateutil.parser.parse(
                                        session_start_match.group(1),
                                        ignoretz=True)
            return None

        sender = util.simplify_SN(matcher.group(2))
        storage = None # which person's list to store this line (IM) in
        ##### determine sender
        for (aliases, lines) in zip(self.both_aliases, self.both_lines):
            if sender in aliases:
                storage = lines
                break
        else:
            return None # didn't match either set of aliases
        ##### make sure it's not empty
        processed = util.strip_punctuation(matcher.group(3).lower()).strip()
        if processed == '':
            return None
        ##### parse timestamp. strptime works the vast majority of the time
        raw_timestamp = matcher.group(1)
        try:
            line_time = datetime.datetime.strptime(raw_timestamp,
                                                   STANDARD_TIMESTAMP)
        except ValueError:
            try:
                line_time = dateutil.parser.parse(raw_timestamp, ignoretz=True)
            except ValueError:
                return None # rare edge case from weird pastes, etc
        ##### Compute datetime from timestamp (time) and previous (date)
        timestamp = datetime.datetime.combine(self.previous.date(),
                                                line_time.time())
        # handle conversations that span multiple days
        if util.crossed_day(self.previous, line_time):
            timestamp += ONE_DAY

        if timestamp < self.previous:
            return None

        ##### Store the timestamp and word counters for the message text
        storage['line_dates'].append(timestamp)
        storage['line_counters'].append(util.make_ngram_counter(self.ngram_N, matcher.group(3)))

        words = util.strip_punctuation(matcher.group(3).lower()).split()
        word_dates = [timestamp] * len(words)
        # intervals = [0] * len(words)
        # intervals[0] = (timestamp - self.previous).total_seconds()

        storage['words'].extend(words)
        storage['word_dates'].extend(word_dates)
        #storage['intervals'].extend(intervals)

        self.previous = timestamp



class PidginReader(LogReader):
    """
    Reads logs from pidgin (IM program).
    """

    def __init__(self, loghome, both_aliases, both_screennames, ngram_N,
                 protocol='jabber'):
        LogReader.__init__(self, loghome, both_aliases, both_screennames, ngram_N)
        self.protocol = protocol
        self.always_check_session = False

        self.session_start_matcher = pidgin_session_start

    def read_all_files(self):
        folder = os.path.join(self.loghome,
                              self.protocol,
                              self.both_screennames[0],
                              self.both_screennames[1])
        try:
            log_files = os.listdir(folder)
        except OSError:
            #print("Warning: Couldn't find pidgin folder %s" % folder)
            return
        for log_file in sorted(log_files):
            self.read_file(os.path.join(folder, log_file))

    @classmethod
    def add_name_and_alias(cls, alias_dict, element):
        """
        Takes an alias dict (mapping names -> list of aliases)
        and a pidgin *.xml buddy/contact element, and extracts the
        name and aliases from the element, storing them in the dict
        """
        name = element.find('name').text.split('/')[0]
        aliases = map(lambda elem: util.simplify_SN(elem.text),
                    element.findall('alias'))
        alias_dict[name].extend(aliases)


    @classmethod
    def make_buddy_aliases(cls, blist_files):
        """
        Takes a list of pidgin's "blist.xml" files, and computes all
        aliases for all buddies.
        """
        buddy_aliases = collections.defaultdict(lambda : [])
        for blist_file in blist_files:
            buddy_list = ET.parse(blist_file).getroot().getchildren()[0]
            # a contact can consist of multiple buddies
            contacts = filter(lambda elem: elem.tag == 'contact',
                            util.list_of_children(buddy_list))
            # a buddy has a screenname & aliases (& other stuff we don't need)
            buddies = filter(lambda elem: elem.tag == 'buddy',
                            util.list_of_children(contacts))
            for b in buddies:
                cls.add_name_and_alias(buddy_aliases, b)
        return buddy_aliases

    @classmethod
    def make_my_aliases(cls, accounts_files):
        """
        Takes a list of pidgin's "account.xml" files, and computes
        all aliases you've used for yourself. Note that these files
        store your password in plain text (that's pidgin's doing).
        None of this code reads or uses your password at all, but
        if you are concerned about privacy please read over this function
        carefully!
        """
        my_aliases = collections.defaultdict(lambda :  [])

        for accounts_file in accounts_files:
            accounts = ET.parse(accounts_file).getroot().getchildren()
            for account in accounts:
                cls.add_name_and_alias(my_aliases, account)
        return my_aliases

class TrillianReader(LogReader):
    """
    Reads logs from the Trillian IM program. Based on the log format
    from ~2005; may have changed since then. I don't know where/how
    to find aliases for Trillian, so I never implemented them.
    """

    def __init__(self, loghome, both_aliases, both_screennames, ngram_N):
        LogReader.__init__(self, loghome, both_aliases, both_screennames, ngram_N)
        self.always_check_session = True

        self.session_start_matcher = trillian_session_start
        return

    def read_all_files(self):
        log_file = os.path.join(self.loghome,
                                self.both_screennames[1]+'.log')

        try:
            self.read_file(log_file)
        except IOError:
            #print("Warning: Couldn't find trillian file %s" % log_file)
            pass # TODO: maybe have a 'require_files_exist' option?

