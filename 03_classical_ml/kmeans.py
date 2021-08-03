# kmeans - Unsupervised Clustering ----
iris = datasets.load_iris()
X = iris.data # Sepal and pedal negth and width
y = iris.target
import sklearn
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, n_jobs = 4, random_state=21)
km.fit(X)

# Graphical Representation ----
import matplotlib.pyplot as plt
new_labels = km.labels_
fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow', edgecolor='k', s=150)
axes[1].scatter(X[:,0], X[:,1], c=new_labels, cmap='jet', edgecolor='k', s=150)
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k',
labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k',
labelsize=20) axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)
plt.show()

# - Identify the best k-value ----
distortions = []
K = range(1,10)
for k in K:
kmeanModel = KMeans(n_clusters=k).fit(X)
kmeanModel.fit(X)
distortions.append(kmeanModel.inertia_)
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Visualize the Dendogram ----
# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))
# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity =
'euclidean'
, linkage = 'ward')
# save clusters for chart
y_hc = hc.fit_predict(points)

from scipy.cluster.hierarchy import dendrogram, linkage
# generate the linkage matrix
Z = linkage(X, 'average')
# set cut-off to 50
max_d = 7.08
# max_d as in max_distance
plt.figure(figsize=(25, 10))
plt.title('Iris Hierarchical Clustering Dendrogram')
plt.xlabel('Species') plt.ylabel('distance')
dendrogram( Z, truncate_mode='lastp',# show only the last p merged
clusters p=10, # Try changing values of p leaf_rotation=90., # rotates
the x axis labels leaf_font_size=8., # font size for the x axis labels)
plt.axhline(y=max_d, c='k')
plt.show()
