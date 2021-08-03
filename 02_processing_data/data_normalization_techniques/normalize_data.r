# - Normalize Data ----
library(datasets)
data(iris)
data = iris 
if (!require('caret')) {
  install.packages("caret", dependencies = TRUE)
  library(caret)
}
#"center", ~ subtracts the mean of the predictor's data (again from the data in x) from the predictor values.
#"scale", ~ divides by the standard deviation.
preProClean <- preProcess(x = data, method = c("scale", "center"))
data <- predict(preProClean, data %>% na.omit)
print(str(data))


# More Advanced Techniques Including Feature Selection ----
#"BoxCox", ~ The Box-Cox transformation was developed for transforming the response variable while another method,...
# ... the Box-Tidwell transformation, was created to estimate transformations of predictor data...
# .... However, the Box-Cox method is simpler, more computationally efficient and is equally effective for estimating power transformations. 
#"YeoJohnson", - The Yeo-Johnson transformation is similar to the Box-Cox model but can accommodate predictors with zero and/or negative...
# ... values (while the predictors values for the Box-Cox transformation must be strictly positive). 
#"expoTrans", ~ The exponential transformation of Manly (1976) can also be used for positive or negative data.
#"range", ~ transformation scales the data to be within rangeBounds. If new samples have values larger or smaller than...
# ... those in the training set, values will be outside of this range.
#"knnImpute", ~ k-nearest neighbor imputation is carried out by finding the k closest samples (Euclidian distance) in the training set. 
#"bagImpute", ~  Imputation via bagging fits a bagged tree model for each predictor (as a function of all the others). 
#"medianImpute", 
#"pca", 
#"ica", 
#"spatialSign", 
#"corr", ~ seeks to filter out highly correlated predictors.
#"zv", ~ identifies numeric predictor columns with a single value (i.e. having zero variance) and excludes them from further calculations.
#"nzv", ~ does the same by applying nearZeroVar exclude "near zero-variance" predictors. The options freqCut and uniqueCut can be...
# ... used to modify the filter.
#"conditionalX" ~ "conditionalX" examines the distribution of each predictor conditional on the outcome...
# ... If there is only one unique value within any class, the predictor is excluded from further calculations...
# .... (see checkConditionalX for an example). When outcome is not a factor, this calculation is not executed. 
# ..... This operation can be time consuming when used within resampling via train.