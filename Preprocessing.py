import os

import pandas
from matplotlib import pyplot as plt


class Preprocessing:

    def __init__(self):
        pass

    def preElaborationData(self, X):
        """
        For each independent variable analyze the distribution of the values

        :param X: training set
        """
        return pandas.DataFrame.describe(X)

    def removeFeatureMinEqualMax(self, X, dataDescription):
        """
        removes features where the minimum value is equal to the maximum value

        :param X: training set
        :param dataDescription: analysis of the distribution of values
        """
        minValues = dataDescription.loc['min']
        maxValues = dataDescription.loc['max']
        featuresNames = dataDescription.columns
        for min, max, feature_name in zip(minValues, maxValues, featuresNames):
            if min == max:
                X = X.drop(feature_name, axis=1)

        return X

    def classDistribution(self, Y):
        """
        verify if the distibution of class in Y is balanced

        :param Y: target
        """
        malware = 0
        goodwere = 0
        for classValue in Y.values:
            if classValue == 1:
                malware += 1

            if classValue == 0:
                goodwere += 1

        print(f'\nclass distribution:'
              f'\nmalwere: {malware} \ngoodwere: {goodwere}')

    def preBoxPlotAnalisys(self, X, Y):
        """
        plot the boxplot of each column of X grouped with respect to Y

        :param X: training set
        :param Y: target
        """
        if not os.path.exists('boxplots'):
            os.makedirs('boxplots')

        X['Label'] = Y['Label']
        for featureX in X.columns:
            X.boxplot(column=featureX, by='Label')
            plt.savefig('boxplots/' + featureX + '.png')
            plt.close()
