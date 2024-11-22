#!/usr/env/bin python
# -*- coding: utf-8 -*-
"""
@author:
@since: 2023/03/09
@evaluation.py
@function: evaluation
"""
from sklearn.metrics import roc_auc_score

auc = roc_auc_score


def test(y_true, y_pred):
    print("AUC: %.4f" % (auc(y_true, y_pred)))

    return auc(y_true, y_pred)
