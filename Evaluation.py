import warnings

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report


class Evaluation:

    def confusionMatrix(self, yTrue, yPredict):
        print(confusion_matrix(yTrue, yPredict))

    def classificationReport(self, yTrue, yPredict):
        print(classification_report(yTrue, yPredict))

    def votingClassifierKfoldCrossValidation(self, X_train, Y_train, X_test,Y_test, decisionTree, randomForest, knn, voting):
        warnings.filterwarnings('ignore')
        # Creating the VotingClassifier with untrained classifiers
        print("\nvoting classifiers...")
        voting_clf = VotingClassifier(estimators=[('dt', decisionTree), ('rf', randomForest), ('knn', knn)], voting=voting)

        commonColumns = list(
            set(X_train.columns) & set(X_test.columns))
        # VotingClassifier training and evaluation on each fold
        voting_clf.fit(X_train.loc[:, commonColumns], Y_train.to_numpy().ravel())
        # score
        print(voting_clf.score(X_test.loc[:, commonColumns], Y_test))

