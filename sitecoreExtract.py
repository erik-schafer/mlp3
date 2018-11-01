#etl
import xml.etree.ElementTree as ET
import json

#explore
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn.metrics as sm
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
#from sklearn.cluster import 
    
import numpy as np
import numpy.random
 #learn
import pandas as pd
import numpy as np

# Only needed if you want to display your plots inline if using Notebook
# change inline to auto if you have Spyder installed
%matplotlib inline


########################
#    ETL
########################
print("begin etl ")

tree = ET.parse("C:/Users/erik.schafer/Documents/sitecore_extract.xml")
root = tree.getroot()

#NLP Payload extraction
nlpElements = root.findall("phrase[@fieldid='NLP Payload']")
data = []
for nlpItem in nlpElements:
    #nlpItem : ['path', 'key', 'itemid', 'fieldid', 'updated']
    nlpData = list(nlpItem)[0].text
    nlpData = json.loads(nlpData)
    #nlpData : ['usage', 'language', 'keywords', 'entities', 'emotion', 'concepts', 'categories']
    if 'keywords' in nlpData:
        keywords = nlpData['keywords']
        keywords = list(map(lambda x: (x['text'], x['relevance']), keywords))
    if 'entities' in nlpData:
        entities = nlpData['entities']
        entities = list(map(lambda x: (x['text'], x['relevance']), entities))
    if 'concepts' in nlpData:
        concepts = nlpData['concepts']
        concepts = list(map(lambda x: (x['text'], x['relevance']), concepts))
    if 'categories' in nlpData:
        categories = nlpData['categories']
        categories = list(map(lambda x: (x['label'], x['score']), categories))
    result = keywords + entities + concepts + categories
    #data.append({'id': nlpItem.get('itemid'), 'features': result})
    data.append(( nlpItem.get('itemid'),  result))

#now we have a list of the following objects:
##{'id': <some guid>, 'features': [{<label>:<relevance>}]}
# [("guid", [("term", .758),("term2", .213),...])]

#count the occurences of all of the labels
counts = {} #want [("term", 5), ("term2", 7), ...]
for item in data:
    features = item[1]
    for k, v in features:
        if k not in counts:
            counts[k] = 1
        else:
            counts[k] = counts[k] + 1

# below is some debug output
#sortedDict = sorted(counts.items(), key = lambda x: x[1], reverse=True)
#print(filter(lambda x: x[1] > 10, sortedDict))

# let's drop all keys that have only a handful of content associated with them
counts = {k: v for k, v in counts.items() if v > 50}

# what we finally want is:
# [id1, id2, id3, ... idn]
# [[term1, term2, term3, ..., termn], [term1, term2, term3, ..., termn], ...]
# [[relevance1, relevance2, .. relevancen], [relevance1, relevance2, .. relevancen], ...]
dataKeys = counts.keys()
ids = []
dataValues = []
for d in data:
    ids.append(d[0])
    dataValues.append([dict(d[1])[x] if x in dict(d[1]) else 0 for x in dataKeys])
    #dataValues.append([d['features'][x] if x in d['features'] else 0 for x in dataKeys])

print("end etl ")
########################################################################################
def graphWithColors(data, labels):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    is3d = False
    if(np.shape(data)[1] > 2):
        is3d = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title("3d scatter")
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black is reserved
            col = [0, 0, 0, 1]
        k_data = []
        for i, d in enumerate(data):
            if(labels[i] == k):
                k_data.append(d)
        k_data = np.array(k_data)
        if is3d:
            ax.scatter(k_data[:, 0], k_data[:, 1], k_data[:,2], c=tuple(col))
        else:
            plt.plot(k_data[:, 0], k_data[:, 1], 'o', markerfacecolor=tuple(col), markersize = 3)
    plt.show()
    if not is3d:
        i = 0
    else:
        
        # NOT implemented
        
        ax.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2], c = 'b', marker='o')
        plt.show()

########################################################################################

########################
#	EXPLORE
########################
print("begin explore ")

x = pd.DataFrame(dataValues, columns = dataKeys)

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Fit_transform scaler to 'X'
X_norm = scaler.fit_transform(x)

# Fit pca to 'X'
pca.fit(X_norm)

print("pca fit..." )

# Plot the explained variances
features = range(0, pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(X_norm)

# Transform the scaled samples: pca_features
pca_features = pca.transform(X_norm)

# Print the shape of pca_features
print(pca_features.shape)

plt.scatter(pca_features[:, 0], pca_features[:, 1])
plt.show()

# what's a good k for our k means cluster?
ks = range(1, 8)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(pca_features)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
########################################################################################
################################
# MORE THOUGHTFUL IMPLEMENTATION
################################

#hyperparams:
TARGET_DIMENSIONS = 27
NUM_CLUSTERS = 5

x = pd.DataFrame(dataValues, columns = dataKeys)

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Fit_transform scaler to 'X'
X_norm = scaler.fit_transform(x)
X_norm = pd.DataFrame(X_norm, columns = dataKeys)

# Fit pca to 'X'
pca.fit(X_norm)

print("pca fit..." )

##########################################################################
# Plot the explained variances
features = range(0, pca.n_components_)

plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.title("PCA before fit")
plt.show()
##########################################################################

# Create a PCA model 
pca = PCA(n_components=TARGET_DIMENSIONS)

# Fit the PCA instance to the scaled samples
pca.fit(X_norm)

# Transform the scaled samples: pca_features
pca_features = pca.transform(X_norm)

# Print the shape of pca_features
print(pca_features.shape)


##########################################################################
# Plot the explained variances
features = range(0, pca.n_components_)
#features = range(0, 50)

plt.bar(features, pca.explained_variance_)
#plt.bar(features, pca.explained_variance_[0:50])

plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.title("Chosen PCA Dimensions")
plt.show()
###
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate the values

# Plot the values
plt.title("3d scatter")
ax.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2], c = 'b', marker='o')
plt.show()

plt.title("2d scatter with depth")
plt.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2])
plt.show()

##################################################
#
#    K Means
#
##################################################

# what's a good k for our k means cluster?
ks = range(1, 10)
inertias = []
silhouette = []
chs = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(pca_features)
    
    # print graphs
    print("k = " + str(k))
    graphWithColors(pca_features, model.labels_)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    if k < 2:
        silhouette.append(0)
        chs.append(0)
    else:
        silhouette.append(metrics.silhouette_score(pca_features, model.labels_, metric='euclidean'))
        chs.append(metrics.calinski_harabaz_score(pca_features, model.labels_))
        
print("# K-means cluster count (k) selection guides: ")
# Plot ks vs inertias
plt.title("Inertais as a function of k")
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# silhouette coefficient
plt.title("Silhouette as a function of k")
plt.plot(ks, silhouette, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('silhouette')
plt.xticks(ks)
plt.show()

# chs coefficient
plt.title("calinski_harabaz_score as a function of k")
plt.plot(ks, chs, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('score')
plt.xticks(ks)
plt.show()


##################################################
#
#    Mean Shift
#
##################################################
#pca_features
model = MeanShift()
model.fit(pca_features)

print("# Mean Shift performance: ")

silhouette = metrics.silhouette_score(pca_features, model.labels_, metric='euclidean')
chs = metrics.calinski_harabaz_score(pca_features, model.labels_)
print("silhouette: {0}".format(silhouette))
print("calinski_harabaz_score: {0}".format(chs))
graphWithColors(pca_features, model.labels_)
##################################################
#
#    DBSCAN
#
##################################################
model = DBSCAN()
model.fit(pca_features)

print("\n# DBSCAN performance: ")

silhouette = metrics.silhouette_score(pca_features, model.labels_, metric='euclidean')
chs = metrics.calinski_harabaz_score(pca_features, model.labels_)
print("silhouette: {0}".format(silhouette))
print("calinski_harabaz_score: {0}".format(chs))
graphWithColors(pca_features, model.labels_)
##################################################
#
#    Ward
#
##################################################
################
# BUILD OUR MODEL
################

model = KMeans(n_clusters = NUM_CLUSTERS)
# QUESTION: pca features or normalized features?
model.fit(pca_features)
#model.fit(X_norm)
labels = model.labels_
# lets get the id and the label for all of the results...
results = [(ids[i], labels[i]) for i in range(len(labels))]

graphWithColors(pca_features, model.labels_)

######################################################################
# FINALLY let's ask some questions of the data:
######################################################################

# todo: what were the terms used in the PCA dimensionality reduction?
# todo: for each category what are the most common tags, or what is the sum of relevances for each tag, or something like that?
# todo: how big is each category?




# what are the urls for classified content?
# what were the most relevant terms which described them?

# data is a collection of ('{guid}', int) tuples
sorted_raw_data = [(d[0], sorted([(k,v) for k,v in d[1]], key = lambda x: float(x[1]), reverse = True)) for d in data]
url = "http://oos.local/sitecore/"


for i in range(NUM_CLUSTERS):
    cluster_elements = list(filter(lambda x: x[1] == i, results))
    examples = [cluster_elements[r] for r in np.random.random_integers(0, len(cluster_elements) - 1, 5)]
    example_data = [next(d[1] for d in sorted_raw_data if d[0] == e[0]) for e in cluster_elements]
    sum_relevance = {}
    frequencies = {}
    
    top5Tags = [next(d[1] for d in sorted_raw_data if d[0] == e[0])[0:5] for e in examples]
    top5Tags = [[pair[0] for pair in item] for item in top5Tags]
    
    
    print("Category {0} - n={1}: ".format(i, len(cluster_elements)))
    #print("5 most frequent features:" + str(frequencies[0:5]))
    #print("5 most relevant features:" + str(sum_relevance[0:5]))
    for index, e in enumerate(examples):
        print(url+e[0] + " " + str(top5Tags[index]))
        #list(map(lambda x: print(url+x[0]), examples))
    print("\n")

print("##############################################")
print(" Mean shift")
model = MeanShift()
model.fit(pca_features)
#sorted_raw_data = [(d[0], sorted([(k,v) for k,v in d[1]], key = lambda x: float(x[1]), reverse = True)) for d in data]
url = "http://oos.local/sitecore/"
labels = model.labels_
results = [(ids[i], labels[i]) for i in range(len(labels))]

for i in range(len(set(labels))):
    cluster_elements = list(filter(lambda x: x[1] == i, results))
    # print(i, len(cluster_elements))
    examples = [cluster_elements[r] for r in np.random.random_integers(0, len(cluster_elements)-1, 5)]
    example_data = [next(d[1] for d in sorted_raw_data if d[0] == e[0]) for e in cluster_elements]
    sum_relevance = {}
    frequencies = {}
    
    top5Tags = [next(d[1] for d in sorted_raw_data if d[0] == e[0])[0:5] for e in examples]
    top5Tags = [[pair[0] for pair in item] for item in top5Tags]
    print("Category {0} - n={1}: ".format(i, len(cluster_elements)-1))
    #print("5 most frequent features:" + str(frequencies[0:5]))
    #print("5 most relevant features:" + str(sum_relevance[0:5]))
    for index, e in enumerate(examples):
        print(url+e[0] + " " + str(top5Tags[index]))
        #list(map(lambda x: print(url+x[0]), examples))
    print("\n")
