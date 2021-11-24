# Import Data ---- 
data(iris)
data = iris
write.csv(data,"data/original.csv", row.names = FALSE)
log_print(paste0("Data Imported..."))
log_print(str(data))