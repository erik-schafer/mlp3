from loadData import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score

import numpy as np

from plot_kmeans_silhouette_analysis import silhouette_analysis
#############
def graphWithColors(data, labels, title = None):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    is3d = False
    if(np.shape(data)[1] > 2):
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
        for i, d in enumerate(data): #for index, data
            if(labels[i] == k):
                k_data.append(d)
        k_data = np.array(k_data)
        if is3d:
            ax.scatter(k_data[:, 0], k_data[:, 1], k_data[:,2], c=tuple(col))
        else:
            plt.plot(k_data[:, 0], k_data[:, 1], 'o', markerfacecolor=tuple(col), markersize = 3)
    plt.show()

#############
x1 = loadData.loadSitecoreExtract()
x3, y3 = loadData.loadCancer()

datasets = dict()
datasets["sitecore"] = {"data": x1, "name": "sitecore"}
datasets["cancer"] = {"data": x3, "name": "cancer", "label": y3}

for k, dataset in datasets.items():
    scaler = StandardScaler()
    dataset["norm"] = scaler.fit_transform(dataset["data"])

#silhouette_analysis(datasets["sitecore"]["norm"], 2, 10)

# ks = range(2, 10)
# silhouettes = []
# inertias = []

# silhouette_analysis(x1_norm, 2, 10)

# for k in ks:
#     model = KMeans(n_clusters=k)
#     labels = model.fit_predict(x3_norm)
#     silhouettes.append(silhouette_score(x3_norm, labels))
#     inertias.append(model.inertia_)
#     print(str(k) + " done")

# plt.plot(ks, inertias, "-o")
# plt.xlabel('number of clusters, k')
# plt.ylabel('inertia')
# plt.xticks(ks)
# plt.show()

# plt.plot(ks, silhouettes, "-o")
# plt.xlabel("k")
# plt.ylabel("silhouettes")
# plt.xticks(ks)
# plt.show()

# for k, dataset in datasets.items():
#     pca = PCA()
#     pca.fit(dataset["norm"])
#     features = range(0, pca.n_components_)
#     sumvar = sum(pca.explained_variance_)
#     plt.bar(features, pca.explained_variance_/sumvar)
#     plt.xlabel('PCA feature')
#     plt.ylabel('variance')
#     plt.xticks(features)
#     plt.show()
#     flag85 = False
#     flag90 = False
#     accumulatedVariance = 0
#     for i in range(len(pca.explained_variance_)):
#         accumulatedVariance += pca.explained_variance_[i]/sumvar
#         if(accumulatedVariance >= .85 and not flag85):
#             print("85% explained at " + str(i+1) + " components")
#             flag85 = True
#         if(accumulatedVariance >= .9 and not flag90):
#             print("90% explained at " + str(i+1) + " components")
#             flag90 = True

datasets["sitecore"]["k"] = 6
datasets["cancer"]["k"] = 2
datasets["sitecore"]["pca_k"] = 180
datasets["cancer"]["pca_k"] = 7

for k, dataset in datasets.items():
    model = KMeans(n_clusters=dataset["k"])
    dataset["kMeansLabels"] = model.fit_predict(dataset["norm"])
    graphWithColors(dataset["norm"], dataset["kMeansLabels"], str(k) + " kMeans - normalized data")
    if "label" in dataset:
        realLabelCount = len(set(dataset["label"]))
        inferredLabelCount = dataset["k"]
        associations = np.zeros(shape=(realLabelCount, inferredLabelCount))
        for i in range(len(dataset["label"])):
            associations[dataset["label"][i],dataset["kMeansLabels"][i]] += 1
        print(associations)


for k, dataset in datasets.items():
    model = GaussianMixture(n_components=dataset["k"])
    model = model.fit(dataset["norm"])
    dataset["EMLabels"] = model.predict(dataset["norm"])
    graphWithColors(dataset["norm"], dataset["EMLabels"], str(k) + " EMLabels - normalized data")
    if "label" in dataset:
        realLabelCount = len(set(dataset["label"]))
        inferredLabelCount = dataset["k"]
        associations = np.zeros(shape=(realLabelCount, inferredLabelCount))
        for i in range(len(dataset["label"])):
            associations[dataset["label"][i],dataset["EMLabels"][i]] += 1
        print(associations)

for k, dataset in datasets.items():
    pca = PCA()
    pca.fit(dataset["norm"])

    # PCA
    # ICA
    # Randomized Projections
    # "your choice"
