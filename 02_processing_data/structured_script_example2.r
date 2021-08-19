# - [Title] - Structured Data Sample - Compiling What We Have Learned So Far ----
### Note: The data is non-linear with categorical variables, fits classification problems...
# - [] - Import Libraries ----
library(datasets)
library(mice)
library(magrittr)
library(outlieR)
library(NoiseFiltersR)
library(caret)
# - [] - Import a Structured Data Table ----
data(iris)
data = iris
str(data)
# - [] - Subset Variables ----
data = data[,-c(1)]
str(data)
# - [] - Impute Missing NA Values ----
missing = data %>% mice::mice(m=5,maxit=50,meth="sample",seed=500,print = FALSE)
missing <- mice::complete(missing, action=as.numeric(2))
data = na.omit(missing)
print(str(data))
# - [] - Impute Outliers ----
out <- data[,-c(4)]
out = out %>% outlieR::impute(flag = NULL, fill = "mean", 
                                level = 0.1, nmax = NULL,
                                side = NULL, crit = "lof", 
                                k = 5, metric = "euclidean", q = 3)
data = cbind(out,Species = data[,4])
print(str(data))
# - [] - Balance the Data ----
target = c("Species") # choose the target variable...
data[,c(target)] = as.factor(data[,c(target)])
formula = as.formula(paste(target, "~."))
noise = GE(formula, data = data, k = 5, kk = ceiling(5/2))
data = noise$cleanData
print(str(data))
table(data$Species)
# - [] - Normalize the Data ----
preProClean <- preProcess(x = data, method = c("scale", "center"))
data <- predict(preProClean, data %>% na.omit)
print(str(data))
# - [] - Save the Processed Data ----
write.csv(data, "data/processed_s2.csv")
