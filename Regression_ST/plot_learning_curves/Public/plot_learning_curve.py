#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 09:29:04 2020

@author: leonardo
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_curve, auc, make_scorer, roc_auc_score


def function_plot_learning_curve(estimator, features, target, train_sizes, cv, title):
    
    _, axes = plt.subplots(figsize=(8, 5))

    axes.set_title(title)

    axes.set_xlabel("Training examples")
    axes.set_ylabel("MAE")

    
    
    train_sizes, train_scores, validation_scores = learning_curve(
    estimator, features, target, train_sizes =
    train_sizes,
    cv = cv, scoring = 'neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis = 1)
    test_scores_mean = -validation_scores.mean(axis = 1)
    
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_std = np.std(validation_scores, axis=1)


    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="lower left")


#    plt.ylim(0,40)

#def function_plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
#    """
#    Generate 3 plots: the test and training learning curve, the training
#    samples vs fit times curve, the fit times vs score curve.
#
#    Parameters
#    ----------
#    estimator : object type that implements the "fit" and "predict" methods
#        An object of that type which is cloned for each validation.
#
#    title : string
#        Title for the chart.
#
#    X : array-like, shape (n_samples, n_features)
#        Training vector, where n_samples is the number of samples and
#        n_features is the number of features.
#
#    y : array-like, shape (n_samples) or (n_samples, n_features), optional
#        Target relative to X for classification or regression;
#        None for unsupervised learning.
#
#    axes : array of 3 axes, optional (default=None)
#        Axes to use for plotting the curves.
#
#    ylim : tuple, shape (ymin, ymax), optional
#        Defines minimum and maximum yvalues plotted.
#
#    cv : int, cross-validation generator or an iterable, optional
#        Determines the cross-validation splitting strategy.
#        Possible inputs for cv are:
#
#          - None, to use the default 5-fold cross-validation,
#          - integer, to specify the number of folds.
#          - :term:`CV splitter`,
#          - An iterable yielding (train, test) splits as arrays of indices.
#
#        For integer/None inputs, if ``y`` is binary or multiclass,
#        :class:`StratifiedKFold` used. If the estimator is not a classifier
#        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
#
#        Refer :ref:`User Guide <cross_validation>` for the various
#        cross-validators that can be used here.
#
#    n_jobs : int or None, optional (default=None)
#        Number of jobs to run in parallel.
#        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
#        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
#        for more details.
#
#    train_sizes : array-like, shape (n_ticks,), dtype float or int
#        Relative or absolute numbers of training examples that will be used to
#        generate the learning curve. If the dtype is float, it is regarded as a
#        fraction of the maximum size of the training set (that is determined
#        by the selected validation method), i.e. it has to be within (0, 1].
#        Otherwise it is interpreted as absolute sizes of the training sets.
#        Note that for classification the number of samples usually have to
#        be big enough to contain at least one sample from each class.
#        (default: np.linspace(0.1, 1.0, 5))
#    """
#
#    _, axes = plt.subplots(figsize=(8, 5))
#
#    axes.set_title(title)
#    if ylim is not None:
#        axes.set_ylim(*ylim)
#    axes.set_xlabel("Training examples")
#    axes.set_ylabel("MAE")
#
#
#    train_sizes, train_scores, test_scores, fit_times, _ = \
#        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, scoring='neg_mean_absolute_error',
#                       train_sizes=train_sizes, return_times=True)
#    train_scores_mean = np.mean(train_scores, axis=1)
#    train_scores_std = np.std(train_scores, axis=1)
#    test_scores_mean = np.mean(test_scores, axis=1)
#    test_scores_std = np.std(test_scores, axis=1)
#    fit_times_mean = np.mean(fit_times, axis=1)
#    fit_times_std = np.std(fit_times, axis=1)
#
#    # Plot learning curve
#    axes.grid()
#    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                         train_scores_mean + train_scores_std, alpha=0.1,
#                         color="r")
#    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                         test_scores_mean + test_scores_std, alpha=0.1,
#                         color="g")
#    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
#                 label="Training score")
#    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
#                 label="Cross-validation score")
#    axes.legend(loc="lower left")


    return plt
