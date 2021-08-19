# Scatterplot Matrix ----
#install install.packages('GGally')
library(GGally)
library(datasets)
data(iris)
data = iris 
ggpairs(data, title = 'Sample Iris Data')

# Optional Method ----
# install.packages('AppliedPredictiveModeling)
library(AppliedPredictiveModeling)
transparentTheme(trans = .4)
featurePlot(x = data[, 1:4], 
            y = data$Species, 
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))