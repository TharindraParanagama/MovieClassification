import numpy as np
from sklearn.cluster import KMeans

def extractFeatures(filename):
    features=[]
    features = np.array([features])
    for line in file(filename):
        row= line .split(',')
        features=np. append(features,np. array([float(x) for x in row[0:5]]))
    return np.array(features,dtype=int)

filename1="/home/tharindra/PycharmProjects/WorkBench/DataMiningAssignment/LabelingBeforeClustering.csv"
features=extractFeatures(filename1)
features=features.reshape(4149,5)

kmeans = KMeans(n_clusters=3)
kmeans.fit(features)

centroids = kmeans.cluster_centers_
label = kmeans.labels_

for i in range(len(features)):
        #print("coordinate:",features[i], "label:", label[i])
        print (label[i])