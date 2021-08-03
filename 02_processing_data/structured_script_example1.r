# - [Title] - Structured Data Sample - Compiling What We Have Learned So Far ----
### Note: The data is linear with continuous variables, fits regression problems...
# - [] - Import Libraries ----
library(quantmod)
library(readr)
library(mice)
library(magrittr)
library(outlieR)
library(caret)
# - [] - Import a Structured Data Table ----
start <- as.Date(Sys.Date()-(365*5)) # start date. 
end <- as.Date(Sys.Date()) # current date. 
getSymbols("GOOG", src = "yahoo", from = start, to = end) # Feed symbol from s&p500. 
data = GOOG # call xts object into env. 
colnames(data) = c("Open", "High", "Low", "Close", "Volume", "Adjusted") # set column names in xts object. 
str(data)
head(data)
# - [] - Subset Variables ----
data = data[,-c(5,6)]
head(data)
# - [] - Impute Missing NA Values ----
missing = data %>% mice::mice(m=5,maxit=50,meth="sample",seed=500,print = FALSE)
missing <- mice::complete(missing, action=as.numeric(2))
data = na.omit(missing)
print(str(data))
# - [] - Impute Outliers ----
data = data %>% outlieR::impute(flag = NULL, fill = "mean", 
                                level = 0.1, nmax = NULL,
                                side = NULL, crit = "lof", 
                                k = 5, metric = "euclidean", q = 3)
print(str(data))
# - [] - Normalize the Data ----
preProClean <- preProcess(x = data, method = c("scale", "center"))
data <- predict(preProClean, data %>% na.omit)
print(str(data))
# - [] - Save the Processed Data ----
write.csv(data, "data/processed.csv")
