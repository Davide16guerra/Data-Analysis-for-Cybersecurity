import os
import pickle

import numpy as np
import pandas
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import DecisionTree
import Evaluation
import KFoldCrossValidation
import Knn
import Preprocessing
import MutualInfo
import RandomForest
from Pca import Pca


def load(path):
    return pandas.read_csv(path)


def serialization(fileName, data):
    # Specifica il percorso della cartella "best settings"
    folder = "best settings"
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(folder + '\\' + fileName + '.pkl', 'wb') as file:
        pickle.dump(data, file)


def deserialization(fileName):
    folder = "best settings"
    with open(folder + '\\' + fileName, 'rb') as file:
        data = pickle.load(file)
    return data


# -------------------------------------------------------------
# --------------------------- MENU ----------------------------
# -------------------------------------------------------------
chosenClassificationAlgorithm = "-1"
while chosenClassificationAlgorithm not in ["decision tree", "random forest", "knn", "test all"]:
    chosenClassificationAlgorithm = input("choose a classification algorithm:\n" +
                                          "- decision tree\n" +
                                          "- random forest\n" +
                                          "- knn\n" +
                                          "- test all\n" +
                                          ">> ")
    if chosenClassificationAlgorithm not in ["decision tree", "random forest", "knn", "test all"]:
        print("invalid command\n")

chosenFeatureSelection = "-1"
while chosenFeatureSelection not in ["mi", "pca", "no"]:
    chosenFeatureSelection = input("\nchoose a feature selection method:\n" +
                                   "- mi\n" +
                                   "- pca\n" +
                                   "- no\n" +
                                   ">> ")
    if chosenFeatureSelection not in ["mi", "pca", "no"]:
        print("invalid command\n")

if chosenClassificationAlgorithm != "test all":
    chosenConfigurationFile = "-1"
    verifyConfigurationFile = False
    while not verifyConfigurationFile:
        chosenConfigurationFile = input("\nenter a configuration file or write \"no\" for automatic search :\n" +
                                        ">> ")
        if chosenConfigurationFile == "no":
            verifyConfigurationFile = True
        else:
            folder = "best settings"
            filePath = os.path.join(os.getcwd(), folder + '\\' + chosenConfigurationFile)
            if os.path.isfile(filePath):
                verifyConfigurationFile = True
            else:
                print("configuration file not found")
else:
    chosenConfigurationFile = "-1"
    dtConfigurationFile = "-1"
    rfConfigurationFile = "-1"
    knnConfigurationFile = "-1"

    while chosenConfigurationFile not in ["yes", "no"]:
        chosenConfigurationFile = input("\ndo you want to use configuration files? (yes/no):\n" +
                                        ">> ")
        if chosenConfigurationFile not in ["yes", "no"]:
            print("invalid command\n")

    if chosenConfigurationFile != "no":
        verifyConfigurationFile = False
        folder = "best settings"
        while not verifyConfigurationFile:
            dtConfigurationFile = input("\ndecision tree >> ")
            filePath = os.path.join(os.getcwd(), folder + '\\' + dtConfigurationFile)
            if os.path.isfile(filePath):
                verifyConfigurationFile = True
            else:
                print("configuration file not found")

        verifyConfigurationFile = False
        while not verifyConfigurationFile:
            rfConfigurationFile = input("\nrandom forest >> ")
            filePath = os.path.join(os.getcwd(), folder + '\\' + rfConfigurationFile)
            if os.path.isfile(filePath):
                verifyConfigurationFile = True
            else:
                print("configuration file not found")

        verifyConfigurationFile = False
        while not verifyConfigurationFile:
            knnConfigurationFile = input("\nknn >> ")
            filePath = os.path.join(os.getcwd(), folder + '\\' + knnConfigurationFile)
            if os.path.isfile(filePath):
                verifyConfigurationFile = True
            else:
                print("configuration file not found")

TestXPath = 'EmberXTest.csv'
TestYPath = "EmberYTest.csv"
TrainXPath = 'EmberXTrain.csv'
TrainYPath = "EmberYTrain.csv"

# Load data
X = load(TrainXPath)
Y = load(TrainYPath)
XEmberTest = load(TestXPath)
YEmberTest = load(TestYPath)

# analyzing the distribution of the values
preprocessing = Preprocessing.Preprocessing()
dataDescriptionTrain = preprocessing.preElaborationData(X)
dataDescriptionTest = preprocessing.preElaborationData(XEmberTest)
print("\ndata description train:\n", dataDescriptionTrain)
print("\ndata description yesy:\n", dataDescriptionTest)

# feature selection
X = preprocessing.removeFeatureMinEqualMax(X, dataDescriptionTrain)
XEmberTest = preprocessing.removeFeatureMinEqualMax(XEmberTest, dataDescriptionTest)

# # verify if the distibution of class in Y is balanced
# preprocessing.classDistribution(Y)

# # pre-elaborate data with PANDAS
# preprocessing.preBoxPlotAnalisys(X, Y)

mi = MutualInfo.MutualInfo()
if chosenFeatureSelection == "mi":
    # Selection with Mutual Information
    print("\n-------------------------------------------------------------")
    print("------------- Selection with Mutual Information -------------")
    print("-------------------------------------------------------------")
    rank = mi.mutualInfoRank(X, Y)
    print(rank, "\n")

    selectedFeatures = mi.topFeatureSelected(rank, 0.1)
    print("\nselected features: ", selectedFeatures)
    print(len(selectedFeatures), "\n")
    XSelected = X.loc[:, selectedFeatures]  # dataset set selected by mi

if chosenFeatureSelection == "pca":
    # Selection with PCA
    print("\n-------------------------------------------------------------")
    print("-------------------- Selection with PCA ---------------------")
    print("-------------------------------------------------------------")
    principalComponentAnalysis = Pca()

    commonColumns = list(
        set(X.columns) & set(XEmberTest.columns))

    pca, pcaList, explainedVariancePercentage = principalComponentAnalysis.pca(X.loc[:, commonColumns])
    print(explainedVariancePercentage)

    XPCA = principalComponentAnalysis.applyPCA(X.loc[:, commonColumns], pca, pcaList)
    XEmberTestPCA = principalComponentAnalysis.applyPCA(XEmberTest.loc[:, commonColumns], pca, pcaList)

    n = principalComponentAnalysis.NumberOfTopPCSelect(explainedVariancePercentage, 0.99)
    print("Number of Top PC select: ", n)

    # create training set and test set with the selected PCs
    XSelected = XPCA.iloc[:, 1:(n + 1)]  # training set selected by PCA
    print(XSelected.shape, "\n")
    XEmberTest = XEmberTestPCA.iloc[:, 1:(n + 1)]  # training set selected by PCA
    print(XEmberTest.shape, "\n")

if chosenFeatureSelection == "no":
    # no feature selection
    print("\n-------------------------------------------------------------")
    print("------------------- no feature selection --------------------")
    print("-------------------------------------------------------------")
    XSelected = X

# Stratify K-Fold cross validation
fold = 5
kFoldCV = KFoldCrossValidation.KFoldCrossValidation()
ListXTrain, ListXTest, ListYTrain, ListYTest = kFoldCV.stratifiedKfold(XSelected, Y, fold)

if chosenConfigurationFile == "no":
    if chosenClassificationAlgorithm == "decision tree" or chosenClassificationAlgorithm == "test all":
        # adopt the stratified CV to determine the best decision tree configuration on original data
        rank = mi.mutualInfoRank(XSelected, Y)  # recalculation of the rank for the selected features
        minThreshold = 0
        max = 0.0
        for key in rank:
            if key[1] >= max:
                max = key[1]
        print("\nmax threshold: ", max, "\n")

        decisionTree = DecisionTree.DecisionTree()
        stepThreshold = 0.05
        maxThreshold = max  # + stepThreshold
        dTbestCriterion, bestTH, bestN, bestEval = decisionTree.determineDecisionTreekFoldConfiguration(ListXTrain,
                                                                                                        ListXTest,
                                                                                                        ListYTrain,
                                                                                                        ListYTest,
                                                                                                        rank,
                                                                                                        minThreshold,
                                                                                                        maxThreshold,
                                                                                                        stepThreshold)
        serialization(chosenFeatureSelection + "_" + chosenClassificationAlgorithm,
                      (dTbestCriterion, bestTH, bestN, bestEval))
        print('\ndecision tree best settings:', 'Best criterion -> ', dTbestCriterion, '; best MI threshold -> ',
              bestTH,
              '; best N -> ', bestN, '; Best CV F -> ', bestEval)

    if chosenClassificationAlgorithm == "random forest" or chosenClassificationAlgorithm == "test all":
        # adopt the stratified CV to determine the best random forest configuration on original data
        rf = RandomForest.RandomForest()
        rFbestCriterion, bestRandomization, bestBootstrap, bestNTrees, bestEval = (
            rf.determineRandomForestKFoldConfiguration(ListXTrain, ListXTest, ListYTrain, ListYTest))
        serialization(chosenFeatureSelection + "_" + chosenClassificationAlgorithm,
                      (rFbestCriterion, bestRandomization, bestBootstrap, bestNTrees, bestEval))
        print('\nrandom forest best settings:', 'Best criterion -> ', rFbestCriterion, '; best randomization -> ',
              bestRandomization,
              '; best boostrap -> ', bestBootstrap, '; Best n trees-> ', bestNTrees, "; best result-> ", bestEval)

    if chosenClassificationAlgorithm == "knn" or chosenClassificationAlgorithm == "test all":
        knn = Knn.Knn()
        bestK, bestEval = knn.determineKNNkFoldConfiguration(ListXTrain, ListXTest, ListYTrain, ListYTest)
        serialization(chosenFeatureSelection + "_" + chosenClassificationAlgorithm, (bestK, bestEval))
        print('\nknn best settings: Best k-> ', bestK, '; Best result-> ', bestEval)

if chosenConfigurationFile != "no" and (
        chosenClassificationAlgorithm == "decision tree" or chosenClassificationAlgorithm == "test all"):
    if chosenClassificationAlgorithm == "test all":
        chosenConfigurationFile = dtConfigurationFile

    bestSettings = deserialization(chosenConfigurationFile)
    dTbestCriterion = bestSettings[0]
    bestTH = bestSettings[1]
    bestN = bestSettings[2]
    bestEval = bestSettings[3]
    print('\ndecision tree best settings:', 'Best criterion -> ', dTbestCriterion, '; best MI threshold -> ', bestTH,
          '; best N -> ', bestN, '; Best CV F -> ', bestEval)

if chosenConfigurationFile != "no" and (
        chosenClassificationAlgorithm == "random forest" or chosenClassificationAlgorithm == "test all"):
    if chosenClassificationAlgorithm == "test all":
        chosenConfigurationFile = rfConfigurationFile

    bestSettings = deserialization(chosenConfigurationFile)
    rFbestCriterion = bestSettings[0]
    bestRandomization = bestSettings[1]
    bestBootstrap = bestSettings[2]
    bestNTrees = bestSettings[3]
    bestEval = bestSettings[4]
    print('\nrandom forest best settings:', 'Best criterion -> ', rFbestCriterion, '; best randomization -> ',
          bestRandomization,
          '; best boostrap -> ', bestBootstrap, '; Best n trees-> ', bestNTrees, "; best result-> ", bestEval)

if chosenConfigurationFile != "no" and (
        chosenClassificationAlgorithm == "knn" or chosenClassificationAlgorithm == "test all"):
    if chosenClassificationAlgorithm == "test all":
        chosenConfigurationFile = knnConfigurationFile

    bestSettings = deserialization(chosenConfigurationFile)
    bestK = bestSettings[0]
    bestEval = bestSettings[1]
    print('\nknn best settings: Best k-> ', bestK, '; Best result-> ', bestEval)

# Decision Tree training
if chosenClassificationAlgorithm == "decision tree":
    print("\n-------------------------------------------------------------")
    print("------------------ Decision Tree training -------------------")
    print("-------------------------------------------------------------")
    decisionTree = DecisionTree.DecisionTree()

    print("decision tree training...")
    commonColumns = list(
        set(XSelected.columns) & set(XEmberTest.columns))

    test1 = XSelected.loc[:, commonColumns]
    DT = decisionTree.decisionTreeLearner(XSelected.loc[:, commonColumns], Y, dTbestCriterion, 500)

    # Decision tree prediction
    test2 = XEmberTest.loc[:, commonColumns]
    Ypredict = DT.predict(XEmberTest.loc[:, commonColumns])

    # Confusion matrix
    evaluation = Evaluation.Evaluation()
    print("confusion matrix:")
    evaluation.confusionMatrix(YEmberTest, Ypredict)

    # Classification report
    print("\nclassification report:")
    evaluation.classificationReport(YEmberTest, Ypredict)

# Random Forest training
if chosenClassificationAlgorithm == "random forest":
    print("\n-------------------------------------------------------------")
    print("------------------ Random Forest training -------------------")
    print("-------------------------------------------------------------")
    randomForest = RandomForest.RandomForest()

    print("random forest training...")
    commonColumns = list(
        set(XSelected.columns) & set(XEmberTest.columns))

    rf = randomForest.randomForestLearner(XSelected.loc[:, commonColumns], Y, rFbestCriterion, bestRandomization,
                                          bestBootstrap,
                                          bestNTrees)

    # Random forest prediction
    Ypredict = rf.predict(XEmberTest.loc[:, commonColumns])

    # Confusion matrix
    evaluation = Evaluation.Evaluation()
    print("confusion matrix:")
    evaluation.confusionMatrix(YEmberTest, Ypredict)

    # Classification report
    print("\nclassification report:")
    evaluation.classificationReport(YEmberTest, Ypredict)

# Knn training
if chosenClassificationAlgorithm == "knn":
    print("\n-------------------------------------------------------------")
    print("----------------------- KNN training ------------------------")
    print("-------------------------------------------------------------")
    kNearestNeighbors = Knn.Knn()

    print("knn training...")
    commonColumns = list(
        set(XSelected.columns) & set(XEmberTest.columns))

    knn = kNearestNeighbors.knnLearner(XSelected.loc[:, commonColumns], Y, bestK)

    # Random forest prediction
    Ypredict = knn.predict(XEmberTest.loc[:, commonColumns])

    # Confusion matrix
    evaluation = Evaluation.Evaluation()
    print("confusion matrix:")
    evaluation.confusionMatrix(YEmberTest, Ypredict)

    # Classification report
    print("\nclassification report:")
    evaluation.classificationReport(YEmberTest, Ypredict)

if chosenClassificationAlgorithm == 'test all':
    print("\n-------------------------------------------------------------")
    print("-------------------------- Test all -------------------------")
    print("-------------------------------------------------------------")
    dt = DecisionTreeClassifier(criterion=dTbestCriterion, min_samples_split=500)
    rf = RandomForestClassifier(criterion=rFbestCriterion, max_features=bestRandomization,
                                max_samples=bestBootstrap, n_estimators=bestNTrees)
    knn = KNeighborsClassifier(n_neighbors=bestK)

    evaluation = Evaluation.Evaluation()
    evaluation.votingClassifierKfoldCrossValidation(XSelected, Y, XEmberTest, YEmberTest, dt, rf, knn, 'hard')
