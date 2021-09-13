preprocess_class_data <- function(x){
  sub = x
  print(paste0("Imputing Outliers..."))
  out = d1 %>% outlieR::impute(flag = NULL, fill = "mean", 
                               level = 0.1, nmax = NULL,
                               side = NULL, crit = "lof", 
                               k = 5, metric = "euclidean", q = 3)
  sub = cbind(out,x5)
  print(str(sub))
  print(paste0("Balancing the Data..."))
  prep = data.frame(sub)
  target = c("x5") # choose the target variable...
  formula = as.formula(paste(target, "~."))
  noise = GE(formula, data = prep, k = 5, kk = ceiling(5/2))
  prep = noise$cleanData
  str(prep)
  print(paste0("Normalize and Feature Engineer the Data..."))
  preProClean <- preProcess(x = prep, method = c("scale", "center","corr"))
  prep <- predict(preProClean, prep %>% na.omit)
  print(str(prep))
}