# - subset variables ----
library(datasets) # datasets library for sample data...
data(iris) # import into the r enviroment... 
data = iris # store the iris dataframe as 'data' in the r enviroment...
remove = c("Petal.Width")
data = data[, !colnames(data) %in% c(remove)] # conditionally select the variables to remove from the dataframe. 
print(str(data))

subset_var <- function(x,y){
  remove = c(toString(y))
  sub = x[, !colnames(x) %in% c(remove)] # conditionally select the variables to remove from the dataframe. 
  print(str(sub))
}

subset_var(data,'Petal.Width')
