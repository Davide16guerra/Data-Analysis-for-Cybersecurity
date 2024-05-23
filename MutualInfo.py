import numpy as np
from sklearn.feature_selection import mutual_info_classif


class MutualInfo:

    def __init__(self):
        pass

    def mutualInfoRank(self, X, Y):
        """
        measures the dependency between the variables.
        It is equal to zero if and only if two random variables are independent,
        and higher values mean higher dependency.

        :param X: training set
        :param Y: target
        """
        print("\nComputing mutual info ranking...")
        independentList = list(X.columns.values)
        res = dict(zip(independentList,
                       mutual_info_classif(X, np.ravel(Y), discrete_features=False, random_state=42)
                       ))
        sortedX = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
        print("Computing mutual info ranking...completed")
        return sortedX

    def topFeatureSelected(self, rank, threshold):
        """
        returns the top features ranked according to MI with MI>=threshold

        :param rank: feature ranked by mutalinfo
        :param threshold: the minimum accepted value
        """
        selectedFeatures = []
        for key, value in rank:
            if value >= threshold:
                selectedFeatures.append(key)

        return selectedFeatures
