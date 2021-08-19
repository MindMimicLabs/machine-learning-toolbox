# statistical summaries in R ----
library(datasets)
data(iris)
data = iris 
summary(data) #summary function in R
# install.packages('pastecs')
library(pastecs)
stat.desc(quakes[,])
# install.packages('Hmisc')
library(Hmisc)
Hmisc::describe(quakes)
# install.packages('psych')
library(psych)
psych::describe(quakes[,])