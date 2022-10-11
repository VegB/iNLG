"""
Reference: https://github.com/wangpf3/imagine-and-verbalize/blob/main/verbalization_learning/lib/utils/text_evaluation.py
"""

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from bert_score import score as bert_score

from collections import defaultdict
import numpy as np
import json
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk

import utils


def tokenize(dict):
    for key in dict:
        new_sentence_list = []
        for sentence in dict[key]:
            a = ''
            for token in nlp(sentence):
                a += token.text
                a += ' '
            new_sentence_list.append(a.rstrip())
        dict[key] = new_sentence_list

    return dict

def evaluator(gts, res, skip_spice=False, skip_meteor=False):
    eval = {}
    # =================================================
    # Set up scorers
    # =================================================
    gts = tokenize(gts)
    res = tokenize(res)

    # =================================================
    # Set up scorers
    # =================================================
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    if not skip_spice:
        scorers.append((Spice(), "SPICE"))
    if not skip_meteor:
        scorers.append((Meteor(), "METEOR"))

    # =================================================
    # Compute scores
    # =================================================
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                eval[m] = sc
        else:
            eval[method] = score
    return eval

def evaluate_sentence(gen_path, ref_path):
    gts = {}
    res = {}
    with open(ref_path, 'r') as f:
        gts_lines = f.readlines()
    with open(gen_path, 'r') as f:
        res_lines = f.readlines()

    for gts_line, res_line in zip(gts_lines, res_lines):
        sample = json.loads(gts_line.strip())
        generation = json.loads(res_line.strip())
        key = '#'.join(sorted(sample['entities']))
        if key not in gts:
            gts[key] = []
            gts[key].append(sample['text'])
            res[key] = []
            res[key].append(generation['text'])
        else:
            gts[key].append(sample['text'])
    return evaluator(gts, res)

def evaluate_story(gen_path, ref_path):
    gts = {}
    res = {}
    with open(ref_path, 'r') as f:
        gts_lines = f.readlines()
    with open(gen_path, 'r') as f:
        res_lines = f.readlines()

    for gts_line, res_line in zip(gts_lines, res_lines):
        sample = json.loads(gts_line.strip())
        generation = json.loads(res_line.strip())
        key = sample['id']
        gts[key] = []
        gts[key].append(sample['text'])
        res[key] = []
        res[key].append(generation['text'])
    return evaluator(gts, res)


def parse_text_to_tokens(sentence):
    return [t.text for t in nlp(sentence)]

def compute_recall(concepts, prediction, verbose=False):
    concept_token_list = parse_text_to_tokens(concepts)
    cnt = 0.
    for t in concept_token_list:
        if prediction.find(t) != -1:
            cnt += 1

    recall = cnt / len(concept_token_list)
    if verbose:
        print(f'concepts:\t{concepts}\nprediction:\t{prediction}\n'
            f'recall rate = {recall*100:.2f} %')
    return recall

def avg_list(lst):
    if not lst:
        return 0
    return sum(lst) / len(lst)

def compute_concept_recall(concept_list, pred_list):
    recall_list = []
    for concepts, prediction in zip(concept_list, pred_list):
        recall_list.append(compute_recall(concepts, prediction))
    recall_rate = avg_list(recall_list)
    return recall_rate


def compute_bertscore(cand_list, refer_list):
    P_mul, R_mul, F_mul = bert_score(cand_list, refer_list, lang="en", rescale_with_baseline=True)
    return F_mul.mean().item()
    

def compute_mauve(pred_list, ref_list):
    from datasets import load_metric
    mauve = load_metric('mauve')
    pred_list = [utils._process_text(text) for text in pred_list]
    ref_list = [utils._process_text(text) for text in ref_list]
    return mauve.compute(predictions=pred_list, references=ref_list, verbose=False, device_id=0).mauve


def ngram_precook(s, n=4, verbose=False):
    words = s.split()
    repeat_index = defaultdict(int)
    for k in range(1,n+1):  # k gram
        all_k_gram_list = []
        for i in range(len(words)-k+1):  # start index
            ngram = tuple(words[i:i+k])
            all_k_gram_list.append(ngram)
        num_unique_k_gram = len(set(all_k_gram_list))
        if len(all_k_gram_list):
            repeat_index[k] = float(num_unique_k_gram) / len(all_k_gram_list)
        else:  # no k-gram
            repeat_index[k] = 1e-13
        if verbose:
            print(f'{k}-gram:\tindex:\t{repeat_index[k]:.2f}\tall:\t{len(all_k_gram_list)}\tunique:{num_unique_k_gram}')
    return repeat_index


def compute_repetition(text_list, n=4, verbose=False):
    """
    See https://arxiv.org/pdf/2202.06417.pdf Section 4.1.2
    """
    repetition_list = defaultdict(list)
    for text in text_list:
        repeat_score = ngram_precook(text, n=n, verbose=verbose)
        for k in range(1, n+1):
            repetition_list[k].append(repeat_score[k])
    repetition_scores = {k: 1 - np.mean(repetition_list[k]) for k in range(1, n+1)}
    if verbose:
        for k in range(1, n+1):
            print(f'rep-{k}:\t{repetition_scores[k]}\t{repetition_list[k]}\tmean:({np.mean(repetition_list[k])})')
    return repetition_scores


def compute_diversity(repetition_scores):
    """
    See https://arxiv.org/pdf/2202.06417.pdf Section 4.1.2
    """
    diversity = 1.
    for k in range(2, 5):
        diversity *= (1-repetition_scores[k])
    return diversity


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(nltk.ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)


def compute_distinct_n(text_list, n=2):
    distinct_n = {}
    for i in range(1, n+1):
        d = distinct_n_corpus_level(sentences=text_list, n=i)
        distinct_n[i] = d
    return distinct_n


if __name__ == "__main__":
    gts = {"cat#dog#boy": ["The dog is the boy's cat.", "The dog eats the cat of the boy."],
           "apple#tree#boy": ["A boy is picking apples from trees."]}
    res = {"cat#dog#boy": ["The dog is the boy's cat."],
           "apple#tree#boy": ["A boy is picking apples from trees and put them into bags."]}
    print(evaluator(gts, res))