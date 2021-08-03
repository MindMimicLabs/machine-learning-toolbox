# - Impute Outliers ----
library(datasets)
data(iris)
data = iris 
if (!require("outlieR")) {
  remotes::install_github("rushkin/outlieR")
  library(outlieR)
}
if (!require("magrittr")) {
  install.packages("magrittr")
  library(magrittr)
}
data = data %>% outlieR::impute(flag = NULL, fill = "mean", 
                                level = 0.1, nmax = NULL,
                                side = NULL, crit = "lof", 
                                k = 5, metric = "euclidean", q = 3)
print(str(data))

data$Sepal.Length = as.numeric(data$Sepal.Length)
data$Sepal.Width = as.numeric(data$Sepal.Width)
data$Petal.Length = as.numeric(data$Petal.Length)
data$Petal.Width = as.numeric(data$Petal.Width)

