# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from collections import Counter
from nltk.translate import bleu_score as nltkbleu

import re

from typing import (
    List,
    Tuple,
)

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s


def _word_prec_recall_f1_score(pred_items, gold_items) -> Tuple[float, float, float]:
    """
     Compute precision, recall and f1 given a set of gold and prediction items.
     :param pred_items: iterable of predicted values
     :param gold_items: iterable of gold values
     :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def word_f1(pred_label: str, gold_labels: List[str], expose_p_and_r: bool = False) -> float:
    if pred_label is None or gold_labels is None:
        return 0
    g_tokens = normalize_answer(pred_label).split()
    scores = [
        _word_prec_recall_f1_score(g_tokens, normalize_answer(a).split())
        for a in gold_labels
    ]
    max_p, max_r, max_f1 = 0, 0, 0
    for p, r, f1 in scores:
        max_p, max_r, max_f1 = max(max_p, p), max(max_r, r), max(f1, max_f1)
    if expose_p_and_r:
        return max_p, max_r, max_f1
    else:
        return max_f1


def bleu(guess: str, answers: List[str], k: int = 4) -> float:
    # cumulative K-gram BLEU score, 4 by default.
    weights = [1 / k for _ in range(k)]
    score = nltkbleu.sentence_bleu(
        [normalize_answer(a).split(" ") for a in answers],
        normalize_answer(guess).split(" "),
        smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
        weights=weights,
    )
    return score