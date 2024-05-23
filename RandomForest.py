import statistics
import warnings

import numpy as np
from pyparsing import results
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


class RandomForest:

    def __init__(self):
        pass

    def randomForestLearner(self, X, Y, criterion, randomization, bootstrapSize, nTrees):
        """
        build a random forest RF from (X,y) with the specified criterion

        :param X: tarining set
        :param Y: target
        :param c: entropy or gini
        :param randomization: the number of features to consider when looking for the best split
        :param bootstrapSize: the number of samples to draw from X to train each base estimator
        :param nTrees: number of trees
        """
        rf = RandomForestClassifier(criterion=criterion, max_features=randomization,
                                    max_samples=bootstrapSize, n_estimators=nTrees)
        warnings.simplefilter("ignore")
        rf.fit(X, Y)
        return rf

    def determineRandomForestKFoldConfiguration(self, ListX_train, ListX_test, ListY_train, ListY_test):
        bestEval = -1
        """
        determine best configuration with respect to the criterion (gini or entropy)
        randomization (sqrt or log2), bootstrap size (with max_samples
        varying among 0.7, 0.8 and 0.9), number of trees (varying among
        10, 20 and 30 )

         :param ListX_train: list of dataframs for each fold of X used for training
         :param ListY_train: list of dataframs for each fold of Y used for training
         :param ListY_test: list of dataframs for each fold of Y used for testing

         :returns: [bestCriterion, bestRandomization, bestBootstrap, bestNTrees, bestEval]
             - bestCriterion: gini or entropy
             - bestRandomization: the best number of features to consider when looking for the best split
             - bestBootstrap: the best number of samples to draw from X to train each base estimator
             - bestNTrees: the best number of trees
             - bestEval: the max avarage f1-score calculated
         """

        print("\nsearching best settings for random forest...")
        for criterion in ['gini', 'entropy']:
            for randomization in ['sqrt', 'log2']:
                for bootstrapSize in [0.7, 0.8, 0.9]:
                    for nTrees in [10, 20, 30]:
                        results = []
                        for X_train, X_test, Y_train, Y_test in zip(ListX_train, ListX_test, ListY_train, ListY_test):
                            rf = RandomForestClassifier(criterion=criterion, max_features=randomization,
                                                        max_samples=bootstrapSize, n_estimators=nTrees)
                            rf.fit(X_train, Y_train.to_numpy().ravel())
                            Y_predict = rf.predict(X_test)
                            results.append(f1_score(Y_test, Y_predict))

                        if len(results) != 0:
                            avg_result = statistics.mean(results)
                        else:
                            avg_result = 0

                        if avg_result > bestEval:
                            bestCriterion = criterion
                            bestRandomization = randomization
                            bestBootstrap = bootstrapSize
                            bestNTrees = nTrees
                            bestEval = avg_result

                        # display the results of each iteration
                        # print("criterion: ", criterion, ";\trandomization: ", randomization, ";\tboostrap: ",
                        #       bootstrapSize,
                        #       ";\tn trees: ", nTrees, ";\tavg_result: ", avg_result)

        return bestCriterion, bestRandomization, bestBootstrap, bestNTrees, bestEval
