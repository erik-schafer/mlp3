# coding: utf-8

# In[28]:


from loadData import *
from plot_kmeans_silhouette_analysis import silhouette_analysis

import time
import os
from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets as skdataset
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score

from scipy.stats import kurtosis

import numpy as np

SAVE = True
TIMESTRING = str(datetime.now().hour) + str(datetime.now().minute)
BASEPATH_IMAGES = "graphs" + TIMESTRING + "/"
print(BASEPATH_IMAGES)
if SAVE: os.makedirs(BASEPATH_IMAGES)


#############
def graphWithColors(data, labels, title=None, filename =""):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    is3d = False
    if (np.shape(data)[1] > 2):
        is3d = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if title == None:
            plt.title("3d scatter")
        else:
            plt.title(title)
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black is reserved
            col = [0, 0, 0, 1]
        k_data = []
        for i, d in enumerate(data):  # for index, data
            if (labels[i] == k):
                k_data.append(d)
        k_data = np.array(k_data)
        if is3d:
            ax.scatter(k_data[:, 0], k_data[:, 1], k_data[:, 2], c=tuple(col))
        else:
            plt.plot(k_data[:, 0], k_data[:, 1], 'o', markerfacecolor=tuple(col), markersize=3)
    if(SAVE):
        print("saving graphWithColors...." +filename)
        plt.savefig(filename)
    else:
        plt.show()
    
    
def makePlot(d, title, filename):
    fig = plt.figure()
    if d.shape[1] >= 3:
        ax = fig.add_subplot(111, projection='3d')
        plt.title(title)
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], c='b', marker='o')
    elif d.shape[1] == 2:
        plt.title(title)
        plt.scatter(d[:,0], d[:,1], c='b', marker='o')
    elif d.shape[1] == 1:
        plt.scatter(d[:, 0], c='b', marker='o')

    if(SAVE):
        print("saving makePlot...."+filename)
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)

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

##############


# In[55]:


def doKmeans(dataset, k, name, reduction, labels=None):
    model = KMeans(n_clusters=k)
    computedLabels = model.fit_predict(dataset)
    graphWithColors(dataset, computedLabels, name + " k=" + str(k) + " kMeans - " + reduction + " data", BASEPATH_IMAGES+"kmeans-"+name+"-"+reduction+".png")
    if labels is not None:
        k_actual = len(set(labels))
        associations = np.zeros(shape=(k_actual, k))
        for i in range(len(labels)):
            associations[labels[i], computedLabels[i]] += 1
        print(associations)
        print("Score is: " , model.score(dataset))
    return computedLabels


# In[56]:


def doEM(dataset, k, name, reduction, labels=None):
    model = GaussianMixture(n_components=k)
    model.fit(dataset)
    computedLabels = model.predict(dataset)
    graphWithColors(dataset, computedLabels, name + " k=" + str(k) + " EM - " + reduction + " data", BASEPATH_IMAGES+"em-"+name+"-"+reduction+".png")
    if labels is not None:
        k_actual = len(set(labels))
        associations = np.zeros(shape=(k_actual, k))
        for i in range(len(labels)):
            associations[labels[i], computedLabels[i]] += 1
        print(associations)
        print("Score is: " , model.score(dataset))
    return computedLabels


# In[37]:

def doExploratoryAnalysis(raw_data, name, k_0, k_n, dataset=None):
    print(name + " Exploratory Analysis ----------------")
    ks = range(k_0, k_n)
    # silhouette of x_i in the model:
    #   1 if far from other clusters
    #   0 is near decision boundary
    #   -1 if possibly misassigned
    sTitle = BASEPATH_IMAGES + "silhouette_analysis-" + name
    silhouette_analysis(raw_data, ks[0], ks[-1], True, sTitle)
    if SAVE:
        for i in range(ks[-1], ks[0]):
            plt.savefig(sTitle + "-" + str(i) + ".png")
            plt.close()
    else:
        plt.show()


    # PCA Explained Variance analysis
    pca = PCA()
    pca.fit(raw_data)
    features = range(0, pca.n_components_)
    sumvar = sum(pca.explained_variance_)
    fig = plt.figure()
    plt.close()
    plt.bar(features, pca.explained_variance_ / sumvar)
    plt.title(name + " PCA Explained Variance")
    plt.xlabel('PCA features')
    plt.ylabel('Variance')
    plt.xticks(features)
    if SAVE:
        plt.savefig(BASEPATH_IMAGES + "Explained_Variance-" + name + ".png")
    else:
        plt.show()
    flag85 = False
    flag90 = False
    accumulatedVariance = 0
    for i in range(len(pca.explained_variance_)):
        accumulatedVariance += pca.explained_variance_[i] / sumvar
        if (accumulatedVariance >= .85 and not flag85):
            print("85% explained at " + str(i + 1) + " components")
            dataset["pca_k85"] = i + 1
            flag85 = True
        if (accumulatedVariance >= .9 and not flag90):
            print("90% explained at " + str(i + 1) + " components")
            dataset["pca_k90"] = i + 1
            flag90 = True

    # iterate over kmeans models and observe silhouettes, inertias
    silhouettes = []
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k)
        computedLabels = model.fit_predict(raw_data)
        silhouettes.append(silhouette_score(raw_data, computedLabels))
        inertias.append(model.inertia_)
    plt.title(name + " inertia by number of clusters")
    plt.plot(ks, inertias, "-o")
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    if SAVE:
        plt.savefig(BASEPATH_IMAGES + "Inertia-" + name + ".png")
    else:
        plt.show()

    plt.title(name + " silhouettes by number of clusters")
    plt.plot(ks, silhouettes, "-o")
    plt.xlabel("k")
    plt.ylabel("silhouettes")
    plt.xticks(ks)
    if SAVE:
        plt.savefig(BASEPATH_IMAGES + "Silhouettes-" + name + ".png")
    else:
        plt.show()

    # iterate over ICA's and observe the mean kurtosis
    ica_ks = range(1, 30)
    plt.title(name + " mean kurtoses by number of clusters")
    avg_kurtoses = []
    for k in ica_ks:
        ica_reduce = FastICA(n_components=k)
        ICAReduceData = ica_reduce.fit_transform(raw_data)
        avg_kurtoses.append(np.mean(kurtosis(ICAReduceData)))

    plt.plot(ica_ks, avg_kurtoses, "-o")
    plt.xlabel("k")
    plt.ylabel("Mean Kurtosis")
    plt.xticks(ica_ks)
    if SAVE:
        plt.savefig(BASEPATH_IMAGES + "Kurtoses-" + name + ".png")
    else:
        plt.show()


# In[39]:


###############

def loadAndInitializeDatasets():
    x1 = loadData.loadSitecoreExtract()
    x3, y3 = loadData.loadCancer()

    datasets = dict()
    datasets["sitecore"] = {"data": x1, "name": "sitecore"}
    datasets["cancer"] = {"data": x3, "name": "cancer", "label": y3}


    for k, dataset in datasets.items():
        scaler = StandardScaler()
        minMaxScaler = MinMaxScaler((0, 1))
        dataset["norm"] = scaler.fit_transform(dataset["data"])
        dataset["scaled"] = minMaxScaler.fit_transform(dataset["data"])
        dataset["k_0"] = 2
        dataset["k_n"] = 15

    datasets["sitecore"]["k"] = 6
    datasets["cancer"]["k"] = 2
    datasets["sitecore"]["pca_k"] = 127
    datasets["cancer"]["pca_k"] = 2
    #datasets["sitecore"]["ica_k"] = 127
    #datasets["cancer"]["ica_k"] = 2
    return datasets


# In[21]:


def doPCA(dataset, k, name):
    pca = PCA(n_components=k)
    pca.fit(dataset)
    pcaData = pca.transform(dataset)
    makePlot(pcaData, name + ": - PCA reduced to " + str(k) + " dimensions", BASEPATH_IMAGES + "PCA-"+name+"-"+str(k)+".png")
    return pcaData


# In[22]:


def doICA(dataset, k, name):
    ica = FastICA()
    ica_reduce = FastICA(n_components=k)
    ICAData = ica.fit_transform(dataset)
    ICAReduceData = ica_reduce.fit_transform(dataset)
    makePlot(ICAData, name + ": - ICA full", BASEPATH_IMAGES + "ICA-"+name+"-full.png")
    makePlot(ICAReduceData, name + ": - ICA reduced to " + str(k) + " dimensions", BASEPATH_IMAGES + "ICA-"+name+"-"+str(k)+".png")
    return ICAData, ICAReduceData


# In[8]:


def doGRP(dataset, k, name):
    grp = GaussianRandomProjection(n_components=k)
    GRPData = grp.fit_transform(dataset)
    makePlot(GRPData, name + ": - GRP reduced to " + str(k) + " dimensions", BASEPATH_IMAGES + "GRP-"+name+"-"+str(k)+".png")
    return GRPData


# In[26]:


def doTSNE(dataset, k, name):
    pre_reduced_data = dataset
    if(dataset.shape[1] > 15):
        pca = PCA(n_components=15)
        pca.fit(dataset)
        pre_reduced_data = pca.transform(dataset)
    tsne = TSNE(n_components=k)
    TSNEData = tsne.fit_transform(pre_reduced_data)
    makePlot(TSNEData, name + ": - TSNE reduced to " + str(k) + " dimensions", BASEPATH_IMAGES + "TSNE-"+name+"-"+str(k)+".png")
    return TSNEData


# In[10]:


def doVT(dataset, percentile):
    feature_variance = [np.var(dataset[:, i]) for i in range(dataset.shape[1])]
    var = np.percentile(feature_variance, percentile)
    vt = VarianceThreshold(threshold=var)
    vtData = vt.fit_transform(dataset)
    return vtData


# In[ ]:


#####################
datasets = loadAndInitializeDatasets()


# In[38]:


for title, dataset in datasets.items():
    raw = dataset["data"].values
    norm = dataset["norm"]
    scaled = dataset["scaled"]
    k = dataset["k"]
    pca_k = dataset["pca_k"]
    label = dataset["label"] if "label" in dataset else None
    
    doExploratoryAnalysis(raw, title, dataset["k_0"], dataset["k_n"], dataset)
    dataset["kMeansLabels"] = doKmeans(norm, k, title, "norm", label)
    dataset["EMLabels"] = doEM(norm, k, title, "norm", label)
    dataset["pca_data"] = doPCA(norm, pca_k, title)
    dataset["ica_data"], dataset["ica_reduce_data"] = doICA(norm, pca_k, title)
    dataset["grp_data"] = doGRP(norm, pca_k, title)
    dataset["tsne_data"] = doTSNE(norm, 3, title)
    dataset["vt80_data"] = doVT(scaled, 80)
    dataset["vt60_data"] = doVT(scaled, 60)


#######################


# doKmeans(dataset, k, name, reduction,labels = None):
#for k, d in datasets.items():
#    for k2, d2 in d.items():
#        if k2 not in ["name", "k", "pca_k", "kMeansLabels", "EMLabels", "label", "data"]:
#            doKmeans(d2, d["k"], d["name"], k2, (d["label"] if "label" in d else None))
#    print(d.keys())


# In[41]:


# In[57]:



data_keys = ['norm','scaled','pca_data','ica_data','ica_reduce_data','grp_data','tsne_data','vt80_data','vt60_data']

for name, dataset in datasets.items():
    k = dataset["k"]
    labels = dataset["label"] if "label" in dataset else None
    for key in data_keys:
        dataset[key + "kl"] =  doKmeans(dataset[key], k, name, key, labels)
        dataset[key + "el"] = doEM(dataset[key], k, name, key, labels)

def doCancerMLP(key, X, y, mod = None):
    clf = MLPClassifier(
            solver='lbfgs',
            alpha=1e-5,
            hidden_layer_sizes=(20,20,20),
            random_state=1)
    plot_learning_curve(clf, "MLP", X, y)
    if SAVE:
        annotation = 'final avg cross val: ' + str(cross_val_score(clf, X, y).mean())
        plt.annotate(annotation, (0, 0), (0, -30), xycoords='axes fraction', textcoords='offset points', va='top', )
        fname = BASEPATH_IMAGES + "learning_curve-cancer-" + key + ".png"
        if mod is not None:
            fname = BASEPATH_IMAGES + "learning_curve-cancer-" + key +"-"+ mod +".png"
        plt.savefig(fname)
    else:
        plt.show()
    print("MLPClassifier",key, mod, cross_val_score(clf, X, y).mean())

for key in data_keys:
    X = datasets["cancer"][key]
    Xk = np.insert(X, X.shape[1],  dataset[key + "kl"], axis=1)
    Xe = np.insert(X, X.shape[1],  dataset[key + "el"], axis=1)
    y = datasets["cancer"]["label"]
    doCancerMLP(key, X, y)
    doCancerMLP(key, Xk, y, "kMeans")
    doCancerMLP(key, Xe, y, "EM")
