import statistics

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import MutualInfo


class DecisionTree:

    def __init__(self):
        pass

    def decisionTreeLearner(self, X, Y, c, min_sample_split):
        """
        build a decision tree T from (X,y) with the specified criterion

        :param X: tarining set
        :param Y: target
        :param c: entropy or gini
        :param min_sample_split:
        """
        clf = DecisionTreeClassifier(criterion=c, min_samples_split=min_sample_split)
        clf.fit(X, Y)
        print("n nodes:", clf.tree_.node_count)
        print("n leave: ", clf.get_n_leaves())
        return clf

    def poltTree(self, T):
        """
        the plot of the decision tree

        :param T: decision tree
        """
        plt.figure(figsize=(25, 20))
        tree.plot_tree(T, filled=True)
        plt.show()

    def determineDecisionTreekFoldConfiguration(self, ListX_train, ListX_test, ListY_train, ListY_test,
                                                rank, min_threshold, max_threshold, step_threshold):
        """
        determine the best configuration with respect to the criterion
        (gini or entropy) and the feature selection threshold (feature selected according to the
        ranking with a threshold ranging among minThreshold and maxTrashold with step
        stepThreshold)

        :param ListX_train: list of dataframs for each fold of X used for training
        :param ListY_train: list of dataframs for each fold of Y used for training
        :param ListY_test: list of dataframs for each fold of Y used for testing
        :param rank: feature ranked by mutalinfo
        :param min_threshold: min accepted value
        :param max_threshold: max accepted value
        :param step_threshold: increase in each step

        :returns: [bestCriterion, bestTH, bestN, bestEval]
            - bestCriterion: gini or entropy
            - bestTH: the best value of the threshold
            - bestN: the length of the top list that contain the feature selected
            - bestEval: the max avarage f1-score calculated
        """
        bestEval = -1
        thresholds = np.arange(min_threshold, max_threshold, step_threshold)

        print("\nsearching best settings for decision tree...")
        for criterion in ['gini', 'entropy']:

            for threshold in thresholds:
                results = []
                mi = MutualInfo.MutualInfo()
                top_list = mi.topFeatureSelected(rank, threshold)

                for X_train, X_test, Y_train, Y_test in zip(ListX_train, ListX_test, ListY_train, ListY_test):
                    common_columns = list(
                        set(X_train.columns) & set(top_list))  # select the common columns between X_train and top_list
                    if len(common_columns) != 0:    # if there aren't common columns don't include in this iteration X_train in the final average f1_score
                        clf = DecisionTreeClassifier(criterion=criterion, min_samples_split=500)
                        clf.fit(X_train.loc[:, common_columns], Y_train)
                        Y_predict = clf.predict(X_test.loc[:, common_columns])
                        results.append(f1_score(Y_test, Y_predict))

                if len(results) != 0:
                    avg_result = statistics.mean(results)
                else:
                    avg_result = 0

                if avg_result > bestEval:
                    bestCriterion = criterion
                    bestTH = threshold
                    bestN = len(top_list)
                    bestEval = avg_result
                # # display the results of each iteration
                # print('Feature Ranking by MI:', 'criterion', criterion, 'MI threshold', threshold,
                #       'N', len(top_list), 'CV F', avg_result)

        return bestCriterion, bestTH, bestN, bestEval
