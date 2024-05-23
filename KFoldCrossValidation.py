from sklearn.model_selection import StratifiedKFold


class KFoldCrossValidation:

    def __init__(self):
        pass

    def stratifiedKfold(self, X, Y, fold):
        """
        is a variation of k-fold which returns stratified folds:
        each set contains approximately the same percentage of samples
        of each target class as the complete set.

        :param X: training set
        :param Y: target
        :param fold: the n parts into which the dataset is divided
        :return: 4 lists containing n dataframes for each fold
        """
        skf = StratifiedKFold(n_splits=fold)
        skf.get_n_splits(X, Y)
        print(skf)
        StratifiedKFold(n_splits=fold, random_state=42, shuffle=True)
        ListXTrain = []
        ListXTest = []
        ListYTrain = []
        ListYTest = []
        for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
            ListXTrain.append(X.loc[train_index])
            ListXTest.append(X.loc[test_index])
            ListYTrain.append(Y.loc[train_index])
            ListYTest.append(Y.loc[test_index])

        return ListXTrain, ListXTest, ListYTrain, ListYTest
