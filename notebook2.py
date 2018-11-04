from loadData import *

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
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score

import numpy as np

from plot_kmeans_silhouette_analysis import silhouette_analysis

#############
def graphWithColors(data, labels, title=None):
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
    plt.show()
##############

def doKmeans(dataset, k, name, reduction, labels=None):
    model = KMeans(n_clusters=k)
    computedLabels = model.fit_predict(dataset)
    graphWithColors(dataset, computedLabels, name + str(k) + " kMeans - " + reduction + " data")
    if labels is not None:
        k_actual = len(set(labels))
        associations = np.zeros(shape=(k_actual, k))
        for i in range(len(labels)):
            associations[labels[i], computedLabels[i]] += 1
        print(associations)
    return computedLabels

def doEM(dataset, k, name, reduction, labels=None):
    model = GaussianMixture(n_components=k)
    model.fit(dataset)
    computedLabels = model.predict(dataset)
    graphWithColors(dataset, computedLabels, name + str(k) + " kMeans - " + reduction + " data")
    if labels is not None:
        k_actual = len(set(labels))
        associations = np.zeros(shape=(k_actual, k))
        for i in range(len(labels)):
            associations[labels[i], computedLabels[i]] += 1
        print(associations)
    return computedLabels

def doExploratoryAnalysis(dataset, name, k_0, k_n):
    print(name + " Exploratory Analysis ----------------")
    ks = range(k_0, k_n)
    silhouette_analysis(dataset, ks[0], ks[-1])
    silhouettes = []
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k)
        computedLabels = model.fit_predict(dataset)
        silhouettes.append(silhouette_score(dataset, computedLabels))
        inertias.append(model.inertia_)

    pca = PCA()
    pca.fit(dataset)
    features = range(0, pca.n_components_)
    sumvar = sum(pca.explained_variance_)

    plt.bar(features, pca.explained_variance_ / sumvar)
    plt.title(name + " PCA Explained Variance")
    plt.xlabel('PCA features')
    plt.ylabel('Variance')
    plt.xticks(features)
    plt.show()
    flag85 = False
    flag90 = False
    accumulatedVariance = 0
    for i in range(len(pca.explained_variance_)):
        accumulatedVariance += pca.explained_variance_[i] / sumvar
        if (accumulatedVariance >= .85 and not flag85):
            print("85% explained at " + str(i + 1) + " components")
            flag85 = True
        if (accumulatedVariance >= .9 and not flag90):
            print("90% explained at " + str(i + 1) + " components")
            flag90 = True
    plt.title(name + " inertia by number of clusters")
    plt.plot(ks, inertias, "-o")
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()
    plt.title(name + " silhouettes by number of clusters")
    plt.plot(ks, silhouettes, "-o")
    plt.xlabel("k")
    plt.ylabel("silhouettes")
    plt.xticks(ks)
    plt.show()

def make3dplot(d, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    ax.scatter(d[:, 0], d[:, 1], d[:, 2], c='b', marker='o')
    plt.show()

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
        dataset["k_n"] = 10

    datasets["sitecore"]["k"] = 6
    datasets["cancer"]["k"] = 2
    datasets["sitecore"]["pca_k"] = 180
    datasets["cancer"]["pca_k"] = 7
    return datasets


def doPCA(dataset, k, name):
    pca = PCA(n_components=k)
    pca.fit(dataset)
    pcaData = pca.transform(dataset)
    make3dplot(pcaData, name + ": - PCA reduced to " + str(k) + " dimensions")
    return pcaData

def doICA(dataset, k, name):
    ica = FastICA()
    ica_reduce = FastICA(n_components=k)
    ICAData = ica.fit_transform(dataset)
    ICAReduceData = ica_reduce.fit_transform(dataset)
    make3dplot(ICAData, name + ": - ICA full")
    make3dplot(ICAReduceData, name + ": - ICA reduced to " + str(k) + " dimensions")
    return ICAData, ICAReduceData

def doGRP(dataset, k, name):
    grp = GaussianRandomProjection(n_components=k)
    GRPData = grp.fit_transform(dataset)
    make3dplot(GRPData, name + ": - GRP reduced to " + str(k) + " dimensions")
    return GRPData

def doTSNE(dataset, k, name):
    pre_reduced_data = dataset
    if(dataset.shape[1] > 15):
        pca = PCA(n_components=15)
        pca.fit(dataset)
        pre_reduced_data = pca.transform(dataset)
    tsne = TSNE(n_components=k)
    TSNEData = tsne.fit_transform(pre_reduced_data)
    make3dplot(TSNEData, name + ": - GRP reduced to " + str(k) + " dimensions")
    return TSNEData

def doVT(dataset, percentile):
    feature_variance = [np.var(dataset[:, i]) for i in range(dataset.shape[1])]
    var = np.percentile(feature_variance, percentile)
    vt = VarianceThreshold(threshold=var)
    vtData = vt.fit_transform(dataset)
    return vtData

#####################
datasets = loadAndInitializeDatasets()
for title, dataset in datasets.items():
    raw = dataset["data"].values
    norm = dataset["norm"]
    scaled = dataset["scaled"]
    doExploratoryAnalysis(raw, title, dataset["k_0"], dataset["k_n"])
    k = dataset["k"]
    pca_k = dataset["pca_k"]
    label = dataset["label"] if "label" in dataset else None
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



