# Boxplot in R ----
library(datasets)
data(iris)
data = iris 
library(caret)
library(AppliedPredictiveModeling)
featurePlot(x = data[, 1:4], 
            y = data$Species, 
            plot = "box", 
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(4,1 ), 
            auto.key = list(columns = 2))