"Problem 2"

import pandas as pd

# Read data into Python
crime_data = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Dataset_Assignment Clustering/crime_data.csv")

#Exploratory Data Analysis
#Measures of Central Tendency / First moment business decision
#Murder
crime_data.Murder.mean() # '.' is used to refer to the variables within object
crime_data.Murder.median()
crime_data.Murder.mode()

#Assault
crime_data.Assault.mean() 
crime_data.Assault.median()
crime_data.Assault.mode()

#UrbanPop
crime_data.UrbanPop.mean() 
crime_data.UrbanPop.median()
crime_data.UrbanPop.mode()

#Rape
crime_data.Rape.mean() 
crime_data.Rape.median()
crime_data.Rape.mode()

# Measures of Dispersion / Second moment business decision
#Murder
crime_data.Murder.var() # variance
crime_data.Murder.std() #standard deviation
range = max(crime_data.Murder) - min(crime_data.Murder) # range
range

#Assault
crime_data.Assault.var() # variance
crime_data.Assault.std() #standard deviation
range = max(crime_data.Assault) - min(crime_data.Assault) # range
range

#UrbanPop
crime_data.UrbanPop.var() # variance
crime_data.UrbanPop.std() #standard deviation
range = max(crime_data.UrbanPop) - min(crime_data.UrbanPop) # range
range

#Rape
crime_data.Rape.var() # variance
crime_data.Rape.std() #standard deviation
range = max(crime_data.Rape) - min(crime_data.Rape) # range
range

# Third moment business decision
crime_data.Murder.skew() # +ve skew , right skew
crime_data.Assault.skew() # +ve skew , right skew
crime_data.UrbanPop.skew() # -ve skew , left skew
crime_data.Rape.skew() # +ve skew , right skew


#Fourth moment business decision
crime_data.Murder.kurt() # platykurtic
crime_data.Assault.kurt() #platykurtic 
crime_data.UrbanPop.kurt() # platykurtic
crime_data.Rape.kurt() #platykurtic

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np


plt.hist(crime_data.Murder) #histogram
plt.hist(crime_data.Assault)
plt.hist(crime_data.UrbanPop)
plt.hist(crime_data.Rape)

#boxplot
plt.boxplot(crime_data.Murder,1,vert=False) #it has no outliers 
plt.boxplot(crime_data.Assault,1,vert=False) #it has no outliers
plt.boxplot(crime_data.UrbanPop,1,vert=False) # no outliers 
plt.boxplot(crime_data.Rape,1,vert=False) # it has outliers and it is right skew

#Normal Quantile-Quantile Plot

import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
#Murder
stats.probplot(crime_data.Murder, dist='norm',plot=pylab) #pylab is visual representation
#transformation to make workex variable normal
import numpy as np
#Assault
stats.probplot(crime_data.Assault, dist='norm',plot=pylab)
#UrbanPop
stats.probplot(crime_data.UrbanPop, dist='norm',plot=pylab)
#Rape
stats.probplot(crime_data.Rape, dist='norm',plot=pylab)
stats.probplot(np.log(crime_data.Rape),dist="norm",plot=pylab)

###### Data Preprocessing########################################

## import packages
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

crime_data.isna().sum()
crime_data.describe() # f na values then count will decrease
crime_data.info() #data types , #object - categorical data

##################  creating Dummy variables using dummies ###############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create dummy variables on categorcal columns

crime_data.info()
crime_data.head()
for col in crime_data:
    print(crime_data[col].unique())
crime_data['States'].unique()
len(crime_data['States'].unique())

# let's have a look at how many labels each variable has

for col in crime_data.columns:
    print(col, ': ', len(crime_data[col].unique()), ' labels')
crime_data.shape    

# let's examine how many columns we will obtain after one hot encoding these variables
pd.get_dummies(crime_data).shape
#We can observe that from with just 5 categorical features we are getting 53 features with the help of one hot encoding.
# let's find the top 10 most frequent categories for the variable States

crime_data.States.value_counts().sort_values(ascending=False).head(20)

# let's make a list with the most frequent categories of the variable

top_10_labels = [y for y in crime_data.States.value_counts().sort_values(ascending=False).head(10).index]
top_10_labels

# get whole set of dummy variables, for all the categorical variables

def one_hot_encoding_top_x(crime_data, variable, top_x_labels):
    # function to create the dummy variables for the most frequent labels
    # we can vary the number of most frequent labels that we encode
    
    for label in top_x_labels:
        crime_data[variable+'_'+label] = np.where(crime_data[variable]==label, 1, 0)

crime_data = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Dataset_Assignment Clustering/crime_data.csv")
# encode States into the 10 most frequent categories
one_hot_encoding_top_x(crime_data, "States", top_10_labels)
crime_data.head()
crime_data.info()

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std()) 
    return(x)
 
# Normalized data frame (considering the numerical part of data)
crime_data_norm = norm_func(crime_data.iloc[:,[1,2,3,4]]) # we take numeric columns, becuase that binary varibales create problem while clustering
crime_data_norm.describe() # min=0, max=1
crime_data_norm.info()

###################### Outlier Treatment #########
"Rape"
# so we have 1 variables which has outliers
# we leave some binary variable because that is categorical variable and categorical variables has no outliers it's like 0's and 1's (Basically categories)
crime_data.dtypes
crime_data.isna().sum()

# let's find outliers 
"Rape"
sns.boxplot(crime_data.Rape);plt.title('Boxplot');plt.show()

###################### Winsorization #####################################
import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Rape'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
crime_data_t = winsorizer.fit_transform(crime_data[['Rape']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(crime_data_t.Rape);plt.title('Rape');plt.show()

#we see no outiers

#################### Missing Values Imputation ##################################
# check for count of NA'sin each column
crime_data.isna().sum()

# there is no na values

################## Type casting###############################################
#Identify duplicates records in the data
duplicate = crime_data.duplicated()
sum(duplicate)

#there is no duplicate values in the data

# Model Building 
# for creating Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # this is for Dendrogram

z = linkage(crime_data_norm, method = "complete", metric = "euclidean")
z

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying Agglomerative Clustering choosing 3 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = 'euclidean').fit(crime_data_norm)
h_complete.labels_ # labels gives cluster Id's

cluster_labels = pd.Series(h_complete.labels_) # convert into series (proper arranging the numbers)
crime_data.info()
crime_data['clust'] = cluster_labels #that cluster_labels which name as 'clust' is added in last column
crime_data = crime_data.iloc[:,[15,1,2,3,4]] # we take numeric columns, becuase that binary varibales create problem while clustering
crime_data.head()

# Aggregate mean of each cluster
crime_data.iloc[:,1:].groupby(crime_data.clust).mean() # from sat it will calculate

# creating a csv file
crime_data.to_csv("crime_dataoutput1python.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Hierarchical Clustering"
os.chdir(path) # current working directory

# or else by default -  c drive
"import os"
"os.getcwd()"

"We are trying to learn more about Crime rate based on the states of U.S, 
"Basically we are trying to find out the crime rate using clustering. 
"The primary objective is to identify Crime rates segments via clustering and design targeted marketing campaigns for each segment.

"In this clustering:-
"Crime rate1 : 1st group (Highest number)
"Crime rate2 : 2nd group (second Highest number)
"Crime rate3 : 0th group (third Highest number)

# 2nd time - changing k values , k = 5

z = linkage(crime_data_norm, method = "complete", metric = "euclidean")
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

h_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = 'euclidean').fit(crime_data_norm)
h_complete.labels_ # labels gives cluster Id's

cluster_labels = pd.Series(h_complete.labels_) # convert into series (proper arranging the numbers)

crime_data['clust'] = cluster_labels #that cluster_labels which name as 'clust' is added in last column

crime_data = crime_data.iloc[:,[15,1,2,3,4]]
crime_data.head()

# Aggregate mean of each cluster
crime_data.iloc[:,1:].groupby(crime_data.clust).mean() # from sat it will calculate

# creating a csv file
crime_data.to_csv("crime_dataoutput2python.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Hierarchical Clustering"
os.chdir(path) # current working directory

# or else by default -  c drive
"import os"
"os.getcwd()"

"In this clustering:-
"Crime rate1 : 2nd group (Highest number)
"Crime rate2 : 1st group (second Highest number)
"Crime rate3 : 4th group (third Highest number)
"Crime rate4 : 0th group (fourth Highest number)
"Crime rate5 : 3rd group (fifth Highest number)

# 3rd time - changing k values , k = 4 , method = single

z = linkage(crime_data_norm, method = "single", metric = "euclidean")
z
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying Agglomerative Clustering choosing 4 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'single', affinity = 'euclidean').fit(crime_data_norm)
h_complete.labels_ # labels gives cluster Id's

cluster_labels = pd.Series(h_complete.labels_) # convert into series (proper arranging the numbers)

crime_data['clust'] = cluster_labels #that cluster_labels which name as 'clust' is added in last column

crime_data = crime_data.iloc[:,[15,1,2,3,4]]
crime_data.head()

# Aggregate mean of each cluster
crime_data.iloc[:,1:].groupby(crime_data.clust).mean() # from sat it will calculate

# creating a csv file
crime_data.to_csv("crime_dataoutput3python.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Hierarchical Clustering"
os.chdir(path) # current working directory

# or else by default -  c drive
"import os"
"os.getcwd()"

"In this clustering:-
"customer segment1 : 2nd group (Highest number)
"customer segment2 : 1st group (second Highest number)
"customer segment3 : 3rd group (third Highest number)
"customer segment4 : 0th group (fourth Highest number)


"6. Share the benefits/impact of the solution - how or in what way the business (client) gets benefit from the solution provided. "

"On the basis clustering We know more about Crime rate based on the states of U.S, 
"Basically we are trying to find out the crime rate using clustering. 
"The primary objective is to identify Crime rates segments via clustering and design targeted marketing campaigns for each segment.
"so, because of clustering we can eassily figure out which group has the highest crime rate and which has lowest, and we can also see individual variables and know about crime rate "








