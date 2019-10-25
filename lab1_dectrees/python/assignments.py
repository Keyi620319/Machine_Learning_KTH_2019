
from drawtree_qt5 import drawTree
import matplotlib.pyplot as plt
import monkdata as m
import numpy as np
import random
import dtree
import math


def prune_tree(tree, validation_set):
    cur_tree = tree
    while 1:
        alternatives = dtree.allPruned(cur_tree)
        best_acc = dtree.check(cur_tree, validation_set)
        best_alt = cur_tree
        for alt in alternatives:
            alt_acc = dtree.check(alt, validation_set)
            if alt_acc >= best_acc:
                best_acc = alt_acc
                best_alt = alt
        if best_alt == cur_tree:
            return cur_tree
        cur_tree = best_alt


def split_data(datasets, proportion):
    return [partition(d, proportion) for d in datasets]


def partition(data, proportion):
    ldata = list(data)
    random.shuffle(ldata)
    lim = int(len(data) * proportion)
    return ldata[:lim], ldata[lim:]


def errTrees(datasets, testsets):
    trees = [dtree.buildTree(d, m.attributes) for d in datasets]
    return [(1-dtree.check(t, ds), 1-dtree.check(t, ts)) for t, ds, ts in zip(trees, datasets, testsets)]


def entropy(datasets):
    return [dtree.entropy(d) for d in datasets]


def avgGain(datasets):
    return [[dtree.averageGain(d, a) for a in m.attributes] for d in datasets]


def main():
    # Get datasets
    monks = [m.monk1, m.monk2, m.monk3]
    monks_test = [m.monk1test, m.monk2test, m.monk3test]

    # Entropy
    # print(entropy(monks))

    # Information Gain
    # print(avgGain(monks))

    # Eval decision trees
    # print(errTrees(monks, monks_test))

    # Pruning
    n_iter = 1000
    props = np.arange(0.3, 0.9, 0.1)
    monk1_stats = np.zeros((2, len(props)))
    monk3_stats = np.zeros((2, len(props)))
    for i in range(len(props)):
        print("prop: ", props[i])
        monk1_prop_stats = np.zeros(n_iter)
        monk3_prop_stats = np.zeros(n_iter)
        for j in range(n_iter):
            print("-> iter ", j)
            sets = split_data([m.monk1, m.monk3], props[i])
            monk1_tree = prune_tree(dtree.buildTree(sets[0][0], m.attributes), sets[0][1])
            monk1_prop_stats[j] = 1 - dtree.check(monk1_tree, m.monk1test)
            monk3_tree = prune_tree(dtree.buildTree(sets[1][0], m.attributes), sets[1][1])
            monk3_prop_stats[j] = 1 - dtree.check(monk3_tree, m.monk3test)
        monk1_stats[0][i] = np.average(monk1_prop_stats)
        monk1_stats[1][i] = np.std(monk1_prop_stats)
        monk3_stats[0][i] = np.average(monk3_prop_stats)
        monk3_stats[1][i] = np.std(monk3_prop_stats)
    plt.errorbar(props, monk1_stats[0], yerr=monk1_stats[1], uplims=True, lolims=True, label="MONK-1")
    plt.errorbar(props, monk3_stats[0], yerr=monk3_stats[1], uplims=True, lolims=True, label="MONK-3")
    plt.ylabel("Generalisation error")
    plt.xlabel("Training set proportion")
    plt.title("Generalisation error depending on training set proportion")
    plt.legend(loc="upper center")
    plt.show()

if __name__ == "__main__":
    main()