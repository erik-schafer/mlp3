
# coding: utf-8

# In[101]:


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
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score

import numpy as np

from plot_kmeans_silhouette_analysis import silhouette_analysis


# In[66]:


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


# In[90]:


x1 = loadData.loadSitecoreExtract()
x3, y3 = loadData.loadCancer()

datasets = dict()
datasets["sitecore"] = {"data": x1, "name": "sitecore"}
datasets["cancer"] = {"data": x3, "name": "cancer", "label": y3}


# In[91]:


for k, dataset in datasets.items():
    scaler = StandardScaler()
    dataset["norm"] = scaler.fit_transform(dataset["data"])


# In[92]:


datasets["sitecore"]["k"] = 6
datasets["cancer"]["k"] = 2
datasets["sitecore"]["pca_k"] = 180
datasets["cancer"]["pca_k"] = 7


# In[70]:


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


# In[71]:


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


# In[79]:



for k, dataset in datasets.items():
    pca = PCA()
    pca.fit(dataset["norm"])
    features = range(0, pca.n_components_)
    sumvar = sum(pca.explained_variance_)
    plt.bar(features, pca.explained_variance_/sumvar)
    plt.xlabel('PCA feature')
    plt.ylabel('variance')
    plt.xticks(features)
    plt.show()
    flag85 = False
    flag90 = False
    accumulatedVariance = 0
    for i in range(len(pca.explained_variance_)):
        accumulatedVariance += pca.explained_variance_[i]/sumvar
        if(accumulatedVariance >= .85 and not flag85):
            print("85% explained at " + str(i+1) + " components")
            flag85 = True
        if(accumulatedVariance >= .9 and not flag90):
            print("90% explained at " + str(i+1) + " components")
            flag90 = True
    # PCA
    # ICA
    # Randomized Projections
    # "your choice"


# In[86]:


for k, dataset in datasets.items():
    pca = PCA(n_components=dataset["pca_k"])
    pca.fit(dataset["norm"])
    dataset["PCA_data"] = pca.transform(dataset["norm"])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(k + ": - PCA reduced to " + str(dataset["pca_k"]) + " dimensions")
    ax.scatter(dataset["PCA_data"][:, 0], dataset["PCA_data"][:, 1], dataset["PCA_data"][:, 2], c = 'b', marker='o')
    plt.show()


# In[96]:


for k, dataset in datasets.items():
    #ica = FastICA(n_components=3)
    ica = FastICA()
    ica_reduce = FastICA(n_components=dataset["pca_k"])
    dataset["ICA_data"] = ica.fit_transform(dataset["norm"])
    dataset["ICA_reduce_data"] = ica_reduce.fit_transform(dataset["norm"])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(k + ": - ICA reduced to " + str(dataset["pca_k"]) + " dimensions")
    ax.scatter(dataset["ICA_reduce_data"][:, 0], dataset["ICA_reduce_data"][:, 1], dataset["ICA_reduce_data"][:, 2], c = 'b', marker='o')
    plt.show()


# In[100]:


for k, dataset in datasets.items():
    grp = GaussianRandomProjection(n_components=dataset["pca_k"])
    dataset["grp_data"] = grp.fit_transform(dataset["norm"])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(k + ": - GRP reduced to " + str(dataset["pca_k"]) + " dimensions")
    ax.scatter(dataset["grp_data"][:, 0], dataset["grp_data"][:, 1], dataset["grp_data"][:, 2], c = 'b', marker='o')
    plt.show()

