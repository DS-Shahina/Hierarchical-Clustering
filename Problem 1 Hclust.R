# Load the dataset
install.packages("readr")
library(readr)
EWA <- read.csv(file.choose())

str(EWA)
#ID.: int
#Balance: int
#Qual_miles: int
#cc1_miles: int
#cc2_miles: int
#cc3_miles: int
#Bonus_miles: int
#Bonus_trans: int
#Flight_miles_12mo: int
#Flight_trans_12: int
#Days_since_enroll: int
#Award.: int
# All are of integer data types

"3. Data Pre-processing"

attach(EWA)
summary(EWA)
# Handling duplicates

dup <- duplicated(EWA)
dup
EWA1_new <- EWA[!duplicated(EWA),]#checking rows if duplicate is there or not
EWA1_new

# Handling Outliers
# 1st of all we will check which attribute or you can say which variables has outliers 
# Those Variables we will take to perform Outlier Treatment.

boxplot(Balance)
boxplot(Qual_miles)
boxplot(cc1_miles) # no outliers
boxplot(cc2_miles) # we leave this because it is a categorical variable and categorical variables has no outliers it's like 0's and 1's (Basically categories)
boxplot(cc3_miles) # we leave this because it is a categorical variable and categorical variables has no outliers it's like 0's and 1's (Basically categories)
boxplot(Bonus_miles)
boxplot(Bonus_trans)
boxplot(Flight_miles_12mo)
boxplot(Flight_trans_12)
boxplot(Days_since_enroll) # no outliers
boxplot(Award.) # no outliers

# so we have 6 variables which has outliers
# Winsorisation -  Means Replacing or you can say rounding off

#1) Balance
qunt1 <- quantile(EWA$Balance,probs = c(.25,.75))
qunt1 # 25% - 18527.5, 75% - 92404.0 
win1 <- quantile(EWA$Balance,probs = c(.01,.80), na.rm =T) # probability of 1% and 80%, and remove na value.
win1 # 1% = 909.62, 99% = 109312.80  
H <- 1.5*IQR(EWA$Balance, na.rm = T)
H #110814.8
# Outliers defined as obeservations that fall below Q1-1.5IQR and above Q3+1.5IQR
# here we have to figure out that if outliers is there if it's below then Q1-1.5IQR then replace with 1%
# Same thing if outliers is there if it's above then Q3+1.5IQR then replace with 99%
# Q1-1.5IQR < Outlier - then replace with 1%
# Q3+1.5IQR > Outlier - then replace with 99%
EWA$Balance[EWA$Balance<(qunt1[1]-H)] <- win1[1]
EWA$Balance[EWA$Balance>(qunt1[2]+H)] <- win1[2]
Balance_new <- boxplot(EWA$Balance)
Balance_new

# now apply for all variables
#2) Qual_miles
qunt2 <- quantile(EWA$Qual_miles,probs = c(.25,.75))
qunt2
win2 <- quantile(EWA$Qual_miles,probs = c(.01,.85), na.rm =T) 
win2 
A <- 1.5*IQR(EWA$Qual_miles, na.rm = T)
A
EWA$Qual_miles[EWA$Qual_miles<(qunt2[1]-A)] <- win2[1]
EWA$Qual_miles[EWA$Qual_miles>(qunt2[2]+A)] <- win2[2]
Qual_miles_new <- boxplot(EWA$Qual_miles)
Qual_miles_new

#3) Bonus_miles
qunt3 <- quantile(EWA$Bonus_miles,probs = c(.25,.75))
qunt3
win3 <- quantile(EWA$Bonus_miles,probs = c(.01,.90), na.rm =T) # we can change the limits(% so that outliers remove)
win3 
B <- 1.5*IQR(EWA$Bonus_miles, na.rm = T)
B
EWA$Bonus_miles[EWA$Bonus_miles<(qunt3[1]-B)] <- win3[1]
EWA$Bonus_miles[EWA$Bonus_miles>(qunt3[2]+B)] <- win3[2]
Bonus_miles_new <- boxplot(EWA$Bonus_miles)
Bonus_miles_new

#4) Bonus_trans
qunt4 <- quantile(EWA$Bonus_trans,probs = c(.25,.75))
qunt4
win4 <- quantile(EWA$Bonus_trans,probs = c(.15,.90), na.rm =T) 
win4 
D <- 1.5*IQR(EWA$Bonus_trans, na.rm = T)
D
EWA$Bonus_trans[EWA$Bonus_trans<(qunt4[1]-D)] <- win4[1]
EWA$Bonus_trans[EWA$Bonus_trans>(qunt4[2]+D)] <- win4[2]
Bonus_trans_new <- boxplot(EWA$Bonus_trans)
Bonus_trans_new

#5) Flight_miles_12mo
qunt5 <- quantile(EWA$Flight_miles_12mo,probs = c(.25,.75))
qunt5
win5 <- quantile(EWA$Flight_miles_12mo,probs = c(.01,.85), na.rm =T) 
win5 
E <- 1.5*IQR(EWA$Flight_miles_12mo, na.rm = T)
E
EWA$Flight_miles_12mo[EWA$Flight_miles_12mo<(qunt5[1]-E)] <- win5[1]
EWA$Flight_miles_12mo[EWA$Flight_miles_12mo>(qunt5[2]+E)] <- win5[2]
Flight_miles_12mo_new <- boxplot(EWA$Flight_miles_12mo)
Flight_miles_12mo_new

#6) Flight_trans_12
qunt6 <- quantile(EWA$Flight_trans_12,probs = c(.25,.75))
qunt6
win6 <- quantile(EWA$Flight_trans_12,probs = c(.01,.85), na.rm =T) 
win6 
G <- 1.5*IQR(EWA$Flight_trans_12, na.rm = T)
G
EWA$Flight_trans_12[EWA$Flight_trans_12<(qunt6[1]-G)] <- win6[1]
EWA$Flight_trans_12[EWA$Flight_trans_12>(qunt6[2]+G)] <- win6[2]
Flight_trans_12_new <- boxplot(EWA$Flight_trans_12)
Flight_trans_12_new

#now we'll removed columns which have outliers.
#and add new columns which do not have outliers.

EWA_old <- subset(EWA,select = -c(2,3,7,8,9,10))
EWA_old

# now add new variable which is outlier free
EWA_new <- c(EWA_old,Balance_new,Qual_miles_new,Bonus_miles_new,Flight_miles_12mo_new,Flight_trans_12_new)
EWA_new
typeof(EWA_new)

# Check whether these all variables in the data set is normally distributed or not
# Normal Quantile-Quantile Plot

str(EWA)

#Balance
qqnorm(Balance)
qqline(Balance)

qqnorm(log1p(Balance))
qqline(log1p(Balance))

# Qual_miles
qqnorm(Qual_miles)
qqline(Qual_miles)

qqnorm(log(Qual_miles)*tanh(Qual_miles))
qqline(log(Qual_miles)*tanh(Qual_miles))

#Bonus_miles
qqnorm(Bonus_miles)
qqline(Bonus_miles)

qqnorm(log1p(Balance))
qqline(log1p(Balance))

#Bonus_trans
qqnorm(Bonus_trans)
qqline(Bonus_trans)

qqnorm(sqrt(Bonus_trans)*sin(Bonus_trans))
qqline(sqrt(Bonus_trans)*sin(Bonus_trans))

#Flight_miles_12mo
qqnorm(Flight_miles_12mo)
qqline(Flight_miles_12mo)

qqnorm(log1p(Flight_miles_12mo)*log10(Flight_miles_12mo))
qqline(log1p(Flight_miles_12mo)*log10(Flight_miles_12mo))

#Days_since_enroll
qqnorm(Days_since_enroll)
qqline(Days_since_enroll)

qqnorm(sqrt(Days_since_enroll))
qqline(sqrt(Days_since_enroll))

###zero Variance
library(ggplot2)
library(ggthemes)
# Use 'apply' and 'var' functions to
# check for variance on numerical values
apply(EWA, 2, var)

# Check for columns having zero variance
which(apply(EWA, 2, var)==0) # ignore the warnings


###### Normalization
# if we have more 0's and 1's in the data we go to Normalization
# to normalize the data we use custom function 
norm <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}
# remove first column 
EWA_mt <- EWA[,-c(1,4,5,6,12)] # removing binary varibales because it is not good for clustering and also remove unwanted variables
EWA_norm <- as.data.frame(lapply(EWA_mt, norm))
# Normalized data have values ranging between 0 to 1

summary(EWA_norm)

# To apply standardization we have inbuilt function scale
# we use mtcars dataset 
# use scale function 
EWA_scale <- as.data.frame(scale(EWA_mt))

"4. Exploratory Data Analysis (EDA):"
#Measures of Central Tendency / First moment business decision
summary(EWA)
#Balance
mean(Balance)
median(Balance)

# mode function
Mode <- function(x){
  a = unique(x) # x is a vector
  return(a[which.max(tabulate(match(x, a)))])
}

Mode(Balance)

#Qual_miles
mean(Qual_miles)
median(Qual_miles)
Mode(Qual_miles)

#cc1_miles
mean(cc1_miles)
median(cc1_miles)
Mode(cc1_miles)

#Bonus_miles
mean(Bonus_miles)
median(Bonus_miles)
Mode(Bonus_miles)

#Bonus_trans  
mean(Bonus_trans)
median(Bonus_trans)
Mode(Bonus_trans)

#Flight_miles_12mo
mean(Flight_miles_12mo)
median(Flight_miles_12mo)
Mode(Flight_miles_12mo)

#Flight_trans_12
mean(Flight_trans_12)
median(Flight_trans_12)
Mode(Flight_trans_12)

#Days_since_enroll
mean(Days_since_enroll)
median(Days_since_enroll)
Mode(Days_since_enroll)

# Measures of Dispersion / Second moment business decision
#Balance
var(Balance) # variance
sd(Balance) #standard deviation
range <- max(Balance) - min(Balance) # range

#Qual_miles
var(Qual_miles) # variance
sd(Qual_miles) #standard deviation
range <- max(Qual_miles) - min(Qual_miles) # range

#cc1_miles
var(cc1_miles) # variance
sd(cc1_miles) #standard deviation
range <- max(cc1_miles) - min(cc1_miles) # range

#Bonus_miles
var(Bonus_miles) # variance
sd(Bonus_miles) #standard deviation
range <- max(Bonus_miles) - min(Bonus_miles) # range

#Bonus_trans
var(Bonus_trans) # variance
sd(Bonus_trans) #standard deviation
range <- max(Bonus_trans) - min(Bonus_trans) # range

#Flight_miles_12mo
var(Flight_miles_12mo) # variance
sd(Flight_miles_12mo) #standard deviation
range <- max(Flight_miles_12mo) - min(Flight_miles_12mo) # range

#Flight_trans_12
var(Flight_trans_12) # variance
sd(Flight_trans_12) #standard deviation
range <- max(Flight_trans_12) - min(Flight_trans_12) # range

#Days_since_enroll
var(Days_since_enroll) # variance
sd(Days_since_enroll) #standard deviation
range <- max(Days_since_enroll) - min(Days_since_enroll) # range

install.packages("moments")
library(moments)

#Third moment business decision
skewness(Balance) #5.00231
skewness(Qual_miles) #7.509577
skewness(cc1_miles) #0.8572472
skewness(Bonus_miles) # 2.841027
skewness(Bonus_trans) #1.156928
skewness(Flight_miles_12mo) #7.448871
skewness(Flight_trans_12) #5.488402
skewness(Days_since_enroll) #0.1201285

#Fourth moment business decision
kurtosis(Balance) #47.10124
kurtosis(Qual_miles) #70.60325
kurtosis(cc1_miles) #2.250927
kurtosis(Bonus_miles) #16.61195
kurtosis(Bonus_trans) #5.740805
kurtosis(Flight_miles_12mo) #97.64108
kurtosis(Flight_trans_12) #45.92294
kurtosis(Days_since_enroll) #2.032204


#Graphical Representation

#Balance - Right Skew 
barplot(Balance)
dotchart(Balance)
hist(Balance)

#Qual_miles - Right Skew
barplot(Qual_miles)
dotchart(Qual_miles)
hist(Qual_miles)

#Bonus_miles - Right Skew
barplot(Bonus_miles)
dotchart(Bonus_miles)
hist(Bonus_miles)

#Bonus_trans - Right Skew
barplot(Bonus_trans)
dotchart(Bonus_trans)
hist(Bonus_trans)

#Flight_miles_12mo - Right Skew
barplot(Flight_miles_12mo)
dotchart(Flight_miles_12mo)
hist(Flight_miles_12mo)

#Flight_trans_12 - Right skew
barplot(Flight_trans_12)
dotchart(Flight_trans_12)
hist(Flight_trans_12)

#Days_since_enroll
barplot(Days_since_enroll)
dotchart(Days_since_enroll)
hist(Days_since_enroll)

# Probability Distribution
install.packages("UsingR")
library("UsingR")
densityplot(Balance)
densityplot(Qual_miles)
densityplot(cc1_miles)
densityplot(cc2_miles)
densityplot(cc3_miles)
densityplot(Bonus_miles)
densityplot(Bonus_trans)
densityplot(Flight_miles_12mo)
densityplot(Flight_trans_12)
densityplot(Days_since_enroll)
densityplot(Award.)

#5. Model Building 

# Distance matrix
d <- dist(EWA_norm, method = "euclidean")
d

fit <- hclust(d, method = "complete")
fit

#Display dendrogram
plot(fit)
plot(fit, hang = -1)

groups <- cutree(fit, k = 3) # cut tree into 3 clusters
# we can cut as we want but 3 is sufficient
groups # 1 is for 1st cluster, 2 is for 2nd cluster, 3 is for 3rd cluster
typeof(groups)
rect.hclust(fit, k = 5, border = "red")

membership <- as.matrix(groups)
membership
typeof(membership)

final <- data.frame(membership, EWA_mt)

aggregate(EWA_mt, by = list(final$membership), FUN = mean)

library(readr)
write_csv(final, "EWAoutput1R.csv") # or write_xl - in excel formal
# get working directory
getwd()

# When k = 3, cluster into 3 parts:-
"We are trying to learn more about EastWest Airlines' customers                 
based on their flying patterns, earning and use of frequent flyer     
rewards, and use of the airline credit card.
The primary objective is to identify customer segments via clustering       
and design targeted marketing campaigns for each segment.

In this clustering:-
customer segment1 : 3rd group (Highest number)
customer segment2 : 2nd group (second Highest number)
customer segment3 : 1st group (third Highest number)
"
#Conclusion - Clustering is not good for large dataset
# in session option choose set working directory - choose Diectory

# 2nd time - change k values

# Distance matrix
d <- dist(EWA_norm, method = "euclidean")
d

fit1 <- hclust(d, method = "Average")
fit1

#Display dendrogram
plot(fit1)
plot(fit1, hang = -1)
groups <- cutree(fit1, k = 5)
groups
typeof(groups)
rect.hclust(fit, k = 7, border = "red")
membership <- as.matrix(groups)
membership
typeof(membership)

final <- data.frame(membership, EWA_mt)

aggregate(EWA_mt, by = list(final$membership), FUN = mean)

library(readr)
write_csv(final, "EWAoutput2R.csv") # or write_xl - in excel formal
# get working directory
getwd()

# when k = 5 , cluster into 5 parts & clustering method is Average linkage
"In this clustering:-
customer segment1 : 5th group (Highest number)
customer segment2 : 2nd group (second Highest number)
customer segment3 : 3rd group (third Highest number)
customer segment3 : 4th group (third Highest number)
customer segment3 : 1st group (third Highest number)
"

# 3rd time - change k values
# Distance matrix
d <- dist(EWA_norm, method = "euclidean")
d

fit2 <- hclust(d, method = "single")
fit2

#Display dendrogram
plot(fit2)
plot(fit2, hang = -1)
groups <- cutree(fit2, k = 4)
groups
typeof(groups)
rect.hclust(fit2, k = 3, border = "red")
membership <- as.matrix(groups)
membership
typeof(membership)

final <- data.frame(membership, EWA_mt)

aggregate(EWA_mt, by = list(final$membership), FUN = mean)

library(readr)
write_csv(final, "EWAoutput3R.csv") # or write_xl - in excel formal
# get working directory
getwd()

# when k = 4 , cluster into 4 parts & clustering method is single linkage
"In this clustering:-
customer segment1 : 4th group (Highest number)
customer segment2 : 2nd group (second Highest number)
customer segment3 : 1st group (third Highest number)
customer segment3 : 3rd group (third Highest number)
"

"6. Share the benefits/impact of the solution - how or in what way the business (client) gets benefit from the solution provided. "

"On the basis clustering we know more about the EastWest Airlines' customers                 
based on their flying patterns, earning and use of frequent flyer     
rewards, and use of the airline credit card. we know more about them by 
identifying customer segments via clustering       
and design targeted marketing campaigns for each segment.
Now, we can more figure out which segment has more profit and which has less"
