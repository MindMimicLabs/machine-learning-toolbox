#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
if (!require("readr")) {
  install.packages("readr")
  library(readr)
}
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# Access all the UCI Machine Learning Repositories @ https://archive.ics.uci.edu/ml/machine-learning-databases/
# Navigate to a selected repo, copy the url path and paste it into the string below.
# ... make sure to copy the data file name to the end of the url path. 
# ... you may require other data importation methods depending on the type of data you select to access. 
data = read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"), header=TRUE)
