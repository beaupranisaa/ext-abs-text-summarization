from extract_summarizer import * 
from collections import Sequence, Counter
from sys import version_info
import nltk
from nltk.corpus import stopwords

PY3 = version_info[0] == 3


if PY3:
    bytes = bytes
    unicode = str
else:
    bytes = str
    unicode = unicode
string_types = (bytes, unicode,)


class TfDocumentModel(object):
    """Term-Frequency document model (term = word)."""
    def __init__(self, words, tokenizer=None):
        if isinstance(words, string_types) and tokenizer is None:
            raise ValueError(
                "Tokenizer has to be given if ``words`` is not a sequence.")
        elif isinstance(words, string_types):
            words = tokenizer.to_words(words)
        elif not isinstance(words, Sequence):
            raise ValueError(
                "Parameter ``words`` has to be sequence or string with tokenizer given.")

        self._terms = Counter(map(unicode.lower, words))
        self._max_frequency = max(self._terms.values()) if self._terms else 1

    @property
    def magnitude(self):
        """
        Lenght/norm/magnitude of vector representation of document.
        This is usually denoted by ||d||.
        """
        return math.sqrt(sum(t**2 for t in self._terms.values()))

    @property
    def terms(self):
        return self._terms.keys()

    def most_frequent_terms(self, count=0):
        """
        Returns ``count`` of terms sorted by their frequency
        in descending order.
        :parameter int count:
            Max. number of returned terms. Value 0 means no limit (default).
        """
        # sort terms by number of occurrences in descending order
        terms = sorted(self._terms.items(), key=lambda i: -i[1])

        terms = tuple(i[0] for i in terms)
        if count == 0:
            return terms
        elif count > 0:
            return terms[:count]
        else:
            raise ValueError(
                "Only non-negative values are allowed for count of terms.")

    def term_frequency(self, term):
        """
        Returns frequency of term in document.
        :returns int:
            Returns count of words in document.
        """
        return self._terms.get(term, 0)

    def normalized_term_frequency(self, term, smooth=0.0):
        """
        Returns normalized frequency of term in document.
        http://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html
        :parameter float smooth:
            0.0 <= smooth <= 1.0, generally set to 0.4, although some
            early work used the value 0.5. The term is a smoothing term
            whose role is to damp the contribution of the second term.
            It may be viewed as a scaling down of TF by the largest TF
            value in document.
        :returns float:
            0.0 <= frequency <= 1.0, where 0 means no occurrence in document
            and 1 the most frequent term in document.
        """
        frequency = self.term_frequency(term) / self._max_frequency
        return smooth + (1.0 - smooth)*frequency

    def __repr__(self):
        return "<TfDocumentModel %s>" % pformat(self._terms)


class LuhnSummarizer(ExtractSummarizer):
    max_gap_size = 4
    # TODO: better recognition of significant words (automatic)
    significant_percentage = 1
    _stop_words = frozenset()

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document, sentences_count):
        words = self._get_significant_words(document.words)
        return self._get_best_sentences(document.sentences,
            sentences_count, self.rate_sentence, words)

    def _get_significant_words(self, words):
        words = map(self.normalize_word, words)   
        stops = set(stopwords.words('english'))
        self._stop_words = frozenset(stops)
        words = tuple(self.stem_word(w) for w in words if w not in self._stop_words)
        # print("====== word1 ======")
        # print(words)
        # print(len(words))
        model = TfDocumentModel(words)

        # take only best `significant_percentage` % words
        best_words_count = int(len(words) * self.significant_percentage) 
        # print(best_words_count)
        words = model.most_frequent_terms(best_words_count)
        # print("====== word2 ======")
        # print(words)
        # print(len(words))
        # take only words contained multiple times in document
        words = tuple(t for t in words if model.term_frequency(t) > 1)
        # print("====== word3 ======")
        # print(words)
        # print(len(words))
        return words

    def rate_sentence(self, sentence, significant_stems):
        ratings = self._get_chunk_ratings(sentence, significant_stems)
        return max(ratings) if ratings else 0

    def _get_chunk_ratings(self, sentence, significant_stems):
        chunks = []
        NONSIGNIFICANT_CHUNK = [0]*self.max_gap_size

        in_chunk = False
        for order, word in enumerate(sentence.words):
            stem = self.stem_word(word)
            # new chunk
            if stem in significant_stems and not in_chunk:
                in_chunk = True
                chunks.append([1])
            # append word to chunk
            elif in_chunk:
                is_significant_word = int(stem in significant_stems)
                chunks[-1].append(is_significant_word)

            # end of chunk
            if chunks and chunks[-1][-self.max_gap_size:] == NONSIGNIFICANT_CHUNK:
                in_chunk = False

        return tuple(map(self._get_chunk_rating, chunks))

    def _get_chunk_rating(self, chunk):
        chunk = self.__remove_trailing_zeros(chunk)
        words_count = len(chunk)
        assert words_count > 0

        significant_words = sum(chunk)
        if significant_words == 1:
            return 0
        else:
            return significant_words**2 / words_count

    def __remove_trailing_zeros(self, collection):
        """Removes trailing zeroes from indexable collection of numbers"""
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]