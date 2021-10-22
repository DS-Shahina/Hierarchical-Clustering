# Load the dataset
library(readxl)
input <- read_excel(file.choose())
mydata <- input[ , c(1,3:8)] # removing 2nd column
# c(1,3:8) means 1,3,4,5,6,7,8 or [,-2] or [,-c(2)]

summary(mydata)
# what we understand with summary-
# 1) Scale is different
# 2) We can find outliers but comparing mean and median
# 3) Distribution of the data can be identified - Skewness (left skew data)

# we can also do normalization but generally when we create dummy 
#variables that time we prefer normalization.
# We just do standardization here
normalized_data <- scale(mydata[, 2:7]) # Excluding the university name

summary(normalized_data) # where mean = 0

# Distance matrix
d <- dist(normalized_data, method = "euclidean")
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

final <- data.frame(membership, mydata)

aggregate(mydata[, 2:7], by = list(final$membership), FUN = mean)

library(readr)
write_csv(final, "hclustoutput.csv") # or write_xl - in excel formal
# get working directory
getwd()

# in session option choose set working directory - choose Diectory

