# Automaticall Filter Noise From Data and Balance ----
library(datasets)
data(iris)
data = iris 
if (!require("NoiseFiltersR")) {
  install.packages("NoiseFiltersR")
  library(NoiseFiltersR)
}
target = c("Species") # choose the target variable...
data[,c(target)] = as.factor(data[,c(target)])
formula = as.formula(paste(target, "~."))
noise = GE(formula, data = data, k = 5, kk = ceiling(5/2))
data = noise$cleanData
print(str(data))


noise$repIdx
