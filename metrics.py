import math
from collections import defaultdict

import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from underthesea import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def eval_report(ga, pa):
    report = {
        "_".join(metric.__name__.split("_")[1:]): metric(ga, pa)
        for metric in [
            calculate_accuracy,
            calculate_precision,
            calculate_recall,
            calculate_f1,
            calculate_cider,
            calculate_rouge_l,
            calculate_meteor,
            calculate_bleu_1,
            calculate_bleu_2,
            calculate_bleu_3,
            calculate_bleu_4,
        ]
    }
    return report


def precook(s, n=4):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occurring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    """
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    """Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    """
    return precook(test, n)


class CiderScorer(object):
    """CIDEr scorer.
    """

    def __init__(self, refs, test=None, n=4, sigma=6.0, doc_frequency=None, ref_len=None):
        """ singular instance """
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.doc_frequency = defaultdict(float)
        self.ref_len = None

        for k in refs.keys():
            self.crefs.append(cook_refs(refs[k]))
            if test is not None:
                self.ctest.append(cook_test(test[k][0]))
            else:
                self.ctest.append(None)  # lens of refs and ctest have to match

        if doc_frequency is None and ref_len is None:
            # compute idf
            self.compute_doc_freq()
            # compute log reference length
            self.ref_len = np.log(float(len(self.crefs)))
        else:
            self.doc_frequency = doc_frequency
            self.ref_len = ref_len

    def compute_doc_freq(self):
        """
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        """
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                self.doc_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.doc_frequency[ngram]))
                # ngram index
                n = len(ngram) - 1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            """
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            """
            delta = float(length_hyp - length_ref)
            # measure cosine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram, count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n] * norm_ref[n])

                assert (not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self):
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)


class Cider:
    """
    Main Class to compute the CIDEr metric

    """

    def __init__(self, gts=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        self.doc_frequency = None
        self.ref_len = None
        if gts is not None:
            tmp_cider = CiderScorer(gts, n=self._n, sigma=self._sigma)
            self.doc_frequency = tmp_cider.doc_frequency
            self.ref_len = tmp_cider.ref_len

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """
        assert (gts.keys() == res.keys())
        cider_scorer = CiderScorer(gts, test=res, n=self._n, sigma=self._sigma, doc_frequency=self.doc_frequency,
                                   ref_len=self.ref_len)
        return cider_scorer.compute_score()

    def __str__(self):
        return 'CIDEr'


def calculate_rouge_l(ground_truths, predictions):
    scorer = rouge_scorer.RougeScorer(['rougeL'])

    rouge_l_scores = []

    for gt, pred in zip(ground_truths, predictions):
        rouge_l = scorer.score(gt, pred)["rougeL"].fmeasure
        rouge_l_scores.append(rouge_l)

    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    return avg_rouge_l


# BLEU
# BLEU
def calculate_bleu_1(ground_truths, predictions):
    bleu_scores = []
    for gt, pred in zip(ground_truths, predictions):
        cc = SmoothingFunction()
        bleu = sentence_bleu(gt, pred, weights=(1, 0, 0, 0), smoothing_function=cc.method4)
        bleu_scores.append(bleu)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return avg_bleu


def calculate_bleu_2(ground_truths, predictions):
    bleu_scores = []
    for gt, pred in zip(ground_truths, predictions):
        cc = SmoothingFunction()
        bleu = sentence_bleu(gt, pred, weights=(0.5, 0.5, 0, 0), smoothing_function=cc.method4)
        bleu_scores.append(bleu)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return avg_bleu


def calculate_bleu_3(ground_truths, predictions):
    bleu_scores = []
    for gt, pred in zip(ground_truths, predictions):
        cc = SmoothingFunction()
        bleu = sentence_bleu(gt, pred, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=cc.method4)
        bleu_scores.append(bleu)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return avg_bleu


def calculate_bleu_4(ground_truths, predictions):
    bleu_scores = []
    for gt, pred in zip(ground_truths, predictions):
        cc = SmoothingFunction()
        bleu = sentence_bleu(gt, pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method4)
        bleu_scores.append(bleu)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return avg_bleu


# METEOR
def calculate_meteor(ground_truths, predictions):
    meteor_scores = []
    for gt, pred in zip(ground_truths, predictions):
        meteor = single_meteor_score(word_tokenize(gt), word_tokenize(pred))
        meteor_scores.append(meteor)

    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    return avg_meteor


def calculate_cider(ground_truths, predictions):
    ground_truths = {idx: word_tokenize(gt) for idx, gt in enumerate(ground_truths)}
    predictions = {idx: word_tokenize(gt) for idx, gt in enumerate(predictions)}
    return Cider().compute_score(ground_truths, predictions)[0]


def calculate_accuracy(ground_truths, predictions):
    return accuracy_score(ground_truths, predictions)


def calculate_f1(ground_truths, predictions):
    return f1_score(ground_truths, predictions, zero_division=0, average="macro")


def calculate_precision(ground_truths, predictions):
    return precision_score(ground_truths, predictions, zero_division=0, average="macro")


def calculate_recall(ground_truths, predictions):
    return recall_score(ground_truths, predictions, zero_division=0, average="macro")
