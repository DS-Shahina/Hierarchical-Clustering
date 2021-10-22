import pandas as pd
import matplotlib.pyplot as plt

Univ1 = pd.read_excel("C:/Users/Admin/OneDrive/Desktop/University_Clustering.xlsx")

Univ1.describe()
Univ1.info()

Univ = Univ1.drop(["State"], axis=1)

# Normalization function
def norm_func(i):
    x = (i-i.min()) / (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Univ.iloc[:, 1:])
df_norm.describe() # after normalization min=0, max=1

# for creating Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # this is for Dendrogram

z = linkage(df_norm, method = "complete", metric = "euclidean")
z

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying Agglomerative Clustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = 'euclidean').fit(df_norm)
h_complete.labels_ # labels gives cluster Id's

cluster_labels = pd.Series(h_complete.labels_) # convert into series (proper arranging the numbers)

Univ['clust'] = cluster_labels #that cluster_labes which name as 'clust' is added in last column

Univ1 = Univ.iloc[:, [7,0,1,2,3,4,5,6]]
Univ1.head()

# Aggregate mean of each cluster
Univ1.iloc[:, 2:].groupby(Univ1.clust).mean() # from sat it will calculate

# creating a csv file
Univ1.to_csv("UniversityOutput.csv", encoding = "utf-8")

import os
path = "D:\\C DRIVE-SSD DATA backup 15-12-2020\\Desktop\\360digitmg material\\Unsupervised-Hierarchical Clustering"
os.chdir(path) # current working directory

# or else by default -  c drive
"import os"
"os.getcwd()" #get working directory




