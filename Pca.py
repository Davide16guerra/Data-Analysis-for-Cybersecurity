import pandas as pd
from sklearn.decomposition import PCA


class Pca:

    def __init__(self):
        pass

    def pca(self, X):
        """
        Linear dimensionality reduction using Singular Value Decomposition
        of the data to project it to a lower dimensional space

        :param X: training set

        :returns: [pca, pcaList, explainedVariancePercentage]
                - pca: pca: principal components of the training set
                - pcaList: the list of names «pc1», «pc2»,…
                - explainedVariancePercentage: Percentage of variance explained by each of the selected component
        """
        print("Training PCA...")
        pca = PCA(n_components=len(X.columns.values))
        pca.fit(X)

        explainedVariance = pca.explained_variance_ratio_
        print(sum(explainedVariance))

        pcalist = []
        for c in range(len(X.columns.values)):
            v = "pc_" + str(c + 1)
            pcalist.append(v)
        print("Training PCA...completed")

        return pca, pcalist, explainedVariance

    def applyPCA(self, X, pca, pcaList):
        """
        transform the Dataframe using PCA and create a data frame collecting
        the principal components obtained by using transform of PCA on the Dataframe

        :param X: training set
        :param pca: principal components of the training set
        :param pcaList: the list of names «pc1», «pc2»,…
        """
        print("Applying PCA...")
        principalComponentsData = pca.transform(X)
        principalDf = pd.DataFrame(data=principalComponentsData, columns=pcaList)
        print("Applying PCA...completed")

        return principalDf

    def NumberOfTopPCSelect(self, explainedVariance, threshold):
        """
        returns the principal components achieving the sum of variance greater than a threshold

        :param explainedVariance: Percentage of variance explained by each of the selected component
        :param threshold:
        """
        nTopPCSelect = 0
        sum = 0
        for value in explainedVariance:
            sum += value
            if sum < threshold:
                nTopPCSelect += 1
            else:
                break

        return nTopPCSelect
