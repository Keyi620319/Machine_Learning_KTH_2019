#!/usr/bin/python
# coding: utf-8

# Lab 3: Bayes Classifier and Boosting

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random


# Bayes classifier functions


def computePrior(labels, W=None):

    # in: labels - N vector of class labels
    # out: prior - C x 1 vector of class priors

    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))
    for k, c in enumerate(classes):
        idx = np.where(labels == c)[0]
        prior[k] = np.sum(W[idx], axis=0) / np.sum(W, axis=0)

    return prior


def mlParams(X, labels, W=None):

    # in:      X - N x d matrix of N data points
    #     labels - N vector of class labels
    # out:    mu - C x d matrix of class means (mu[i] - class i mean)
    #      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)

    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))
    for k in range(Nclasses):
        # Compute mu
        idx = np.where(labels==classes[k])[0]
        mu[k,:] = np.sum(W[idx,:] * X[idx,:], axis=0) / np.sum(W[idx], axis=0)
        # Compute sigma
        dif = X[idx,:]-mu[k]
        sigma[k] = np.diag(np.sum(np.square(dif) * W[idx], axis=0) / np.sum(W[idx], axis=0))

    return mu, sigma


def classifyBayes(X, prior, mu, sigma):

    # in:      X - N x d matrix of M data points
    #      prior - C x 1 matrix of class priors
    #         mu - C x d matrix of class means (mu[i] - class i mean)
    #      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
    # out:     h - N vector of class predictions for test points

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))
    for k in range(Nclasses):
        dif = X-mu[k]
        lnDetSigma = -0.5*np.log(np.linalg.det(sigma[k]))
        lnPrior = np.log(prior[k])
        for i in range(Npts):
            logProb[k][i] = lnDetSigma - 0.5*np.inner(dif[i]/np.diag(sigma[k]), dif[i]) + lnPrior
    
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h


class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# Boosting functions


def trainBoost(base_classifier, X, labels, T=10):

    # in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
    #                   X - N x d matrix of N data points
    #              labels - N vector of class labels
    #                   T - number of boosting iterations
    # out:    classifiers - (maximum) length T Python list of trained classifiers
    #              alphas - (maximum) length T Python list of vote weights

    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # Train a new classifier
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # Perform classification for each point
        vote = classifiers[-1].classify(X)

        # Compute the weighted error
        classes = np.unique(labels)
        epsilon = 0
        for c in classes:
            idx = np.where(vote == c)[0]
            epsilon += np.sum(np.transpose(wCur[idx]) *  (1 - (c == labels[idx])))
        # Compute alhpa
        alpha = 0.5*(np.log(1-epsilon) - np.log(epsilon))
        alphas.append(alpha)
        # Update the weights
        delta = np.reshape(np.where(vote==labels, -alpha, alpha), (Npts, 1))
        wCur = np.multiply(wCur, np.exp(delta))
        wCur /= np.sum(wCur)
        
    return classifiers, alphas


def classifyBoost(X, classifiers, alphas, Nclasses):

    # in:       X - N x d matrix of N data points
    # classifiers - (maximum) length T Python list of trained classifiers as above
    #      alphas - (maximum) length T Python list of vote weights
    #    Nclasses - the number of different classes
    # out:  yPred - N vector of class predictions for test points

    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        for t in range(Ncomps):
            res = classifiers[t].classify(X)
            for i in range(Npts):
                votes[i,res[i]] += alphas[t]

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)
