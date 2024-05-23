import statistics
import warnings

from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier


class Knn:
    def __init__(self):
        pass

    def knnLearner(self, X, Y, k):
        """
        build a Knn from (X,y) with the specified criterion

        :param X: tarining set
        :param Y: target
        :param k: number of considered neighbors
        """
        knn = KNeighborsClassifier(n_neighbors=k)
        warnings.simplefilter("ignore")
        knn.fit(X, Y)
        return knn

    def determineKNNkFoldConfiguration(self, ListX_train, ListX_test, ListY_train, ListY_test,):
        """
        determine best configuration with respect to the value of k varying between 1, 3.

        :param ListX_train:
        :param ListX_test:
        :param ListY_train:
        :param ListY_test:

        :returns: [bestK, bestEval]
            - bestK: the best number of considered neighbors
            - bestEval: the max avarage f1-score calculated
        """
        warnings.filterwarnings('ignore')
        bestEval = -1

        print("\nsearching best settings for knn...")
        for k in [1, 2, 3]:
            results = []
            for X_train, X_test, Y_train, Y_test in zip(ListX_train, ListX_test, ListY_train, ListY_test):
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, Y_train.to_numpy().ravel())
                Y_predict = knn.predict(X_test)
                results.append(f1_score(Y_test, Y_predict))

            if len(results) != 0:
                avg_result = statistics.mean(results)
            else:
                avg_result = 0

            if avg_result > bestEval:
                bestK = k
                bestEval = avg_result

        return bestK, bestEval