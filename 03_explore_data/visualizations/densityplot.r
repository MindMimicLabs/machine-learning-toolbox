# Density Plot ----
library(datasets)
data(iris)
data = iris 
library(caret)
library(AppliedPredictiveModeling)
transparentTheme(trans = .9)
featurePlot(x = data[, 1:4], 
            y = data$Species,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(4, 1), 
            auto.key = list(columns = 3))