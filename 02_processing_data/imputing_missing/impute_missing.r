# Impute Missing/Na Values ----
library(datasets)
data(iris)
data = iris 
if (!require("mice")) {
  install.packages("mice")
  library(mice)
}
if (!require("magrittr")) {
  install.packages("magrittr")
  library(magrittr)
}
missing = data %>% mice::mice(m=5,maxit=50,meth="sample",seed=500,print = FALSE)
missing <- mice::complete(missing, action=as.numeric(2))
data = na.omit(missing)
write.csv(data,"data.csv")
print(str(data))