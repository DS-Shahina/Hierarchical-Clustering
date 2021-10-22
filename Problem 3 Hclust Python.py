"Problem 3"

import pandas as pd

# Read data into Python
Tcc = pd.read_excel("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Dataset_Assignment Clustering/Telco_customer_churn.xlsx")
Tcc.head()
Tcc.shape
Tcc.columns.values
Tcc.dtypes
Tcc.info()

#  Delete column which is not necessary
# Only 11 columns has numeric values rest of all are categorical values
#let's remove columns which are not neended - we remove binary and categorical columns because it creates problem while clustering
Tcc1 = Tcc[['Number of Referrals','Tenure in Months','Avg Monthly Long Distance Charges','Avg Monthly GB Download','Monthly Charge','Total Charges','Total Refunds','Total Extra Data Charges','Total Long Distance Charges','Total Revenue']]
Tcc1.head()
Tcc1.shape
Tcc1.columns.values
Tcc1.dtypes
Tcc1.info()
Tcc1.describe()
#Exploratory Data Analysis
# Check the descriptive statistics of numeric variables
Tcc1.describe()
#1st moment Business Decision # Measures of Central Tendency / First moment business decision
Tcc1.mean()
Tcc1.median()
Tcc1.mode()

#2nd moment business decision # Measures of Dispersion / Second moment business decision
Tcc1.var() 
Tcc1.std()

#3rd moment Business Decision
Tcc1.skew()

#4th moment Business Decision
Tcc1.kurtosis()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np
import seaborn as sns
#Histogram
for i, predictor in enumerate(Tcc1):
    plt.figure(i)
    sns.histplot(data=Tcc1, x=predictor)
    
#boxplot    
for i, predictor in enumerate(Tcc1):
    plt.figure(i)
    sns.boxplot(data=Tcc1, x=predictor)

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std()) 
    return(x)
 
# Normalized data frame (considering the numerical part of data)
Tcc1_norm = norm_func(Tcc1) # we take numeric columns, because that binary varibales create problem while clustering
Tcc1_norm.describe() # min=0, max=1
Tcc1_norm.info()

###################### Outlier Treatment #########
"Number of Referrals","Avg Monthly GB Download","Total Refunds","Total Extra Data Charges","Total Long Distance Charges","Total Revenue"
# so we have 6 variables which has outliers
# we leave some binary variable because that is categorical variable and categorical variables has no outliers it's like 0's and 1's (Basically categories)

# let's find outliers 
"Number of Referrals"
sns.boxplot(Tcc1['Number of Referrals']);plt.title('Boxplot');plt.show()

###################### Winsorization #####################################
import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Number of Referrals'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Tcc1_t = winsorizer.fit_transform(Tcc1[['Number of Referrals']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Tcc1_t['Number of Referrals']);plt.title('Number of Referrals');plt.show()

#we see no outiers

"Avg Monthly GB Download"
sns.boxplot(Tcc1['Avg Monthly GB Download']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Avg Monthly GB Download'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Tcc1_t = winsorizer.fit_transform(Tcc1[['Avg Monthly GB Download']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Tcc1_t['Avg Monthly GB Download']);plt.title('Avg Monthly GB Download');plt.show()

#we see no outiers

"Total Refunds"
sns.boxplot(Tcc1['Total Refunds']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Total Refunds'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Tcc1_t = winsorizer.fit_transform(Tcc1[['Total Refunds']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Tcc1_t['Total Refunds']);plt.title('Total Refunds');plt.show()

#we see no outiers

"Total Extra Data Charges"
sns.boxplot(Tcc1['Total Extra Data Charges']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Total Extra Data Charges'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Tcc1_t = winsorizer.fit_transform(Tcc1[['Total Extra Data Charges']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Tcc1_t['Total Extra Data Charges']);plt.title('Total Extra Data Charges');plt.show()

#we see no outiers

"Total Long Distance Charges"
sns.boxplot(Tcc1['Total Long Distance Charges']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Total Long Distance Charges'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Tcc1_t = winsorizer.fit_transform(Tcc1[['Total Long Distance Charges']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Tcc1_t['Total Long Distance Charges']);plt.title('Total Long Distance Charges');plt.show()

#we see no outiers

"Total Revenue"
sns.boxplot(Tcc1['Total Revenue']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Total Revenue'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Tcc1_t = winsorizer.fit_transform(Tcc1[['Total Revenue']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Tcc1_t['Total Revenue']);plt.title('Total Revenue');plt.show()

#we see no outiers

#################### Missing Values Imputation ##################################
# check for count of NA'sin each column
Tcc1.isna().sum()

# there is no na values

################## Type casting###############################################
#Identify duplicates records in the data
duplicate = Tcc1.duplicated()
sum(duplicate)

#there is no duplicate values in the data

# Model Building 
# for creating Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # this is for Dendrogram
import gower #pip install gower in console window
a = gower.gower_matrix(Tcc1)
a
z = linkage(Tcc1_norm, method = "complete", metric = "euclidean")
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

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = 'euclidean').fit(Tcc1_norm)
h_complete.labels_ # labels gives cluster Id's

cluster_labels = pd.Series(h_complete.labels_) # convert into series (proper arranging the numbers)
Tcc1.info()
Tcc1['clust'] = cluster_labels #that cluster_labels which name as 'clust' is added in last column
Tcc1 = Tcc1.iloc[:,[10,0,1,2,3,4,5,6,7,8,9]] # we take numeric columns, becuase that binary varibales create problem while clustering
Tcc1.head()

# Aggregate mean of each cluster
Tcc1.iloc[:,1:].groupby(Tcc1.clust).mean() # from sat it will calculate

# creating a csv file
Tcc1.to_csv("telcocustomeroutput1python.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Hierarchical Clustering"
os.chdir(path) # current working directory

# or else by default -  c drive
"import os"
"os.getcwd()"

"Customer churn, also known as customer attrition, 
"occurs when customers stop doing business with a company or stop using a company’s services. 
"By being aware of and monitoring churn rate, companies are equipped to determine their customer retention success rates and identify strategies for improvement. 
"We will use a machine learning model to understand the precise customer behaviors and attributes., 
"Basically we are trying to find out the Customer churn pattern using clustering. 
"The primary objective is to identify Customer churn segments via clustering and design targeted marketing campaigns for each segment.

"In this clustering:-
"Customer Churn1 : 1st group (Highest number)
"Customer Churn2 : 0th group (second Highest number)
"Customer Churn3 : 2nd group (third Highest number)

# 2nd time - changing k values , k = 4

z = linkage(Tcc1_norm, method = "complete", metric = "euclidean")
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

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = 'euclidean').fit(Tcc1_norm)
h_complete.labels_ # labels gives cluster Id's

cluster_labels = pd.Series(h_complete.labels_) # convert into series (proper arranging the numbers)

Tcc1['clust'] = cluster_labels #that cluster_labels which name as 'clust' is added in last column

Tcc1 = Tcc1.iloc[:,[10,0,1,2,3,4,5,6,7,8,9]]
Tcc1.head()

# Aggregate mean of each cluster
Tcc1.iloc[:,1:].groupby(Tcc1.clust).mean() # from sat it will calculate

# creating a csv file
Tcc1.to_csv("telcocustomeroutput2python.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Hierarchical Clustering"
os.chdir(path) # current working directory

# or else by default -  c drive
"import os"
"os.getcwd()"

"In this clustering:-
"Customer Churn1 : 3rd group (Highest number)
"Customer Churn2 : 0th group (second Highest number)
"Customer Churn3 : 1st group (third Highest number)
"Customer Churn3 : 2nd group (third Highest number)

"6. Share the benefits/impact of the solution - how or in what way the business (client) gets benefit from the solution provided. "

"On the basis clustering We know more about Customer churn based on some varibales, 
"Basically we are trying to find out the Customer Churn using clustering. 
"The primary objective is to identify Customer Churn segments via clustering and design targeted marketing campaigns for each segment.
"so, because of clustering we can eassily figure out which group has the highest Customer Churn and which has lowest, and we can also see individual variables and know about Customer Churn"






















