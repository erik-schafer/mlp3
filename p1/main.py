basepath = "C:/workspace/MachineLearning-Project1/data"

from sklearn import datasets, decomposition
from sklearn.datasets import load_iris

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import graphviz 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

import numpy as np

#################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def loadCancer():    
    filename = "breast_cancer_data.csv"
    data = pd.read_csv(basepath + '/' + filename)
    data = data.drop("id", 1)
    data = data.drop("Unnamed: 32", 1)
    X = data.drop("diagnosis", 1)
    y = pd.get_dummies(data["diagnosis"])["M"]
    return X, y

def loadCreditCard():
    filename = "creditcard.csv"
    data = pd.read_csv(basepath + '/' + filename)
    X = data.drop("Class", 1)
    y = data["Class"]
    return X, y

def loadNintendo():
    filename = "vgsales.csv"
    data = pd.read_csv(basepath + '/' + filename)
    data = data.drop("Rank", 1)
    data = data.drop("Name", 1)
    data.Platform = pd.Categorical(data.Platform).codes
    data.Genre = pd.Categorical(data.Genre).codes
    data.Publisher = pd.Categorical(data.Publisher).codes
    data = data.dropna()
    X = data.drop("Platform",1)
    y = data.Platform
    return X, y

X, y = loadCancer()

print("Cancer Data Set")
#############
train_scores, validation_scores = validation_curve(KNeighborsClassifier(), X, y, "n_neighbors", range(1,25))
ts = pd.DataFrame(train_scores)
vs = pd.DataFrame(validation_scores)
ts.mean(1)
fig = plt.figure()
plt.title("Knn - varying n_neighbors")
plt.plot(ts.mean(1))
plt.plot(vs.mean(1))
fig.show()
clf = KNeighborsClassifier(n_neighbors=10)
plot_learning_curve(clf, "KNeighborsClassifier", X, y)
plt.show()
print("KNeighborsClassifier", cross_val_score(clf, X, y).mean())


#############
train_scores, validation_scores = validation_curve(tree.DecisionTreeClassifier(), X, y, "max_depth", range(1,16))
ts = pd.DataFrame(train_scores)
vs = pd.DataFrame(validation_scores)
ts.mean(1)
fig = plt.figure()
plt.title("DecisionTreeClassifier - varying depth")
plt.plot(ts.mean(1))
plt.plot(vs.mean(1))
fig.show()
clf = tree.DecisionTreeClassifier(max_depth=6)
plot_learning_curve(clf, "DecisionTreeClassifier", X, y)
plt.show()
print("DecisionTreeClassifier", cross_val_score(clf, X, y).mean())


#############
#vary something other than hidden layers...
#train_scores, validation_scores = validation_curve(MLPClassifier(
#        solver='lbfgs', 
#        alpha=1e-5, 
#        hidden_layer_sizes=(3,8), 
#        random_state=1), X, y, "hidden_layer_sizes", [(x,y)for x in range(2,10) for y in range(2,10)])
clf = MLPClassifier(
        solver='lbfgs', 
        alpha=1e-5, 
        hidden_layer_sizes=(10,10,10), 
        random_state=1)
plot_learning_curve(clf, "MLP", X, y)
plt.show()
print("MLPClassifier", cross_val_score(clf, X, y).mean())

#############
train_scores, validation_scores = validation_curve(svm.SVC(), X, y, "degree", range(1,5))
ts = pd.DataFrame(train_scores)
vs = pd.DataFrame(validation_scores)
fig = plt.figure()
plt.title("SVC - varying degree")
plt.plot(ts.mean(1))
plt.plot(vs.mean(1))
fig.show()

clf = svm.SVC(degree=3)
print("SVC", cross_val_score(clf, X, y).mean())
#plot_learning_curve(clf, "SVC", X, y)
#plt.show()

#############
## Ensemble method
train_scores, validation_scores = validation_curve(GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=1.0,
        max_depth=1, 
        random_state=0
    ), X, y, "max_depth", range(2,15))
ts = pd.DataFrame(train_scores)
vs = pd.DataFrame(validation_scores)
fig = plt.figure()
plt.title("GradientBoostingClassifier - varying max_depth")
plt.plot(ts.mean(1))
plt.plot(vs.mean(1))
fig.show()

clf = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=1.0,
        max_depth=1, 
        random_state=0
    )
print("GradientBoosting", cross_val_score(clf, X, y).mean())
#plot_learning_curve(clf, "GradientBoostingClassifier", X, y)
#plt.show()


plt.show()
############################################################################################################################################################

#############################################################################################################################################################


X, y = loadNintendo()

print("Video Game Data Set")
train_scores, validation_scores = validation_curve(KNeighborsClassifier(), X, y, "n_neighbors", range(1,15))
ts = pd.DataFrame(train_scores)
vs = pd.DataFrame(validation_scores)
ts.mean(1)
fig = plt.figure()
plt.title("Knn - varying n_neighbors")
plt.plot(ts.mean(1))
plt.plot(vs.mean(1))
fig.show()
clf = KNeighborsClassifier(n_neighbors=8)
plot_learning_curve(clf, "KNeighborsClassifier", X, y)
plt.show()
print("KNeighborsClassifier", cross_val_score(clf, X, y).mean())


#############
train_scores, validation_scores = validation_curve(tree.DecisionTreeClassifier(), X, y, "max_depth", range(1,16))
ts = pd.DataFrame(train_scores)
vs = pd.DataFrame(validation_scores)
ts.mean(1)
fig = plt.figure()
plt.title("DecisionTreeClassifier - varying depth")
plt.plot(ts.mean(1))
plt.plot(vs.mean(1))
fig.show()
clf = tree.DecisionTreeClassifier(max_depth=1)
plot_learning_curve(clf, "DecisionTreeClassifier", X, y)
plt.show()
print("DecisionTreeClassifier", cross_val_score(clf, X, y).mean())


#############
#vary something other than hidden layers...
#train_scores, validation_scores = validation_curve(MLPClassifier(
#        solver='lbfgs', 
#        alpha=1e-5, 
#        hidden_layer_sizes=(3,8), 
#        random_state=1), X, y, "hidden_layer_sizes", [(x,y)for x in range(2,10) for y in range(2,10)])
clf = MLPClassifier(
        solver='lbfgs', 
        alpha=1e-5, 
        hidden_layer_sizes=(20,20,20), 
        random_state=1)
plot_learning_curve(clf, "MLP", X, y)
plt.show()
print("MLPClassifier", cross_val_score(clf, X, y).mean())

#############
#train_scores, validation_scores = validation_curve(svm.SVC(), X, y, "degree", range(1,5))
#ts = pd.DataFrame(train_scores)
#vs = pd.DataFrame(validation_scores)
#fig = plt.figure()
#plt.title("SVC - varying degree")
#plt.plot(ts.mean(1))
#plt.plot(vs.mean(1))
#fig.show()

clf = svm.SVC(degree=3)
print("SVC", cross_val_score(clf, X, y).mean())
#plot_learning_curve(clf, "SVC", X, y)
#plt.show()

#############
## Ensemble method
# train_scores, validation_scores = validation_curve(GradientBoostingClassifier(
#         n_estimators=100, 
#         learning_rate=1.0,
#         max_depth=1, 
#         random_state=0
#     ), X, y, "max_depth", range(2,15))
# ts = pd.DataFrame(train_scores)
# vs = pd.DataFrame(validation_scores)
# fig = plt.figure()
# plt.title("GradientBoostingClassifier - varying max_depth")
# plt.plot(ts.mean(1))
# plt.plot(vs.mean(1))
# fig.show()

clf = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=1.0,
        max_depth=1, 
        random_state=0
    )
print("GradientBoosting", cross_val_score(clf, X, y).mean())
#plot_learning_curve(clf, "GradientBoostingClassifier", X, y)
#plt.show()

x = input("continue...")