# Pre -Process Data ----
# - [] - Subset Variables ----
data = data[,-c(1)]
# - [] - Impute Missing NA Values ----
missing = data %>% mice::mice(m=5,maxit=50,meth="sample",seed=500,print = FALSE)
missing <- mice::complete(missing, action=as.numeric(2))
data = na.omit(missing)
# - [] - Impute Outliers ----
out <- data[,-c(4)]
out = out %>% outlieR::impute(flag = NULL, fill = "mean", 
                              level = 0.1, nmax = NULL,
                              side = NULL, crit = "lof", 
                              k = 5, metric = "euclidean", q = 3)
data = cbind(out,Species = data[,4])
# - [] - Balance the Data ----
target = c("Species") # choose the target variable...
data[,c(target)] = as.factor(data[,c(target)])
formula = as.formula(paste(target, "~."))
noise = GE(formula, data = data, k = 5, kk = ceiling(5/2))
data = noise$cleanData
# - [] - Normalize the Data ----
preProClean <- preProcess(x = data, method = c("scale", "center"))
data <- predict(preProClean, data %>% na.omit)
# - [] - Save the Processed Data ----
assign("data",data, envir = globalenv())
write.csv(data, "data/processed.csv")
log_print(paste0("Data Pre-processed..."))
log_print(str(data))