# - Correlation Analysis in R ----
library(datasets)
data(iris)
data = iris 
cor = cor(data[,c(1:4)]) # inputs must be in numeric data type...
print(cor)