#install.packages('readr')
library(readr) # load the package into the r enviroment...
dir = getwd() # locate and store the set working directory of the current r session
setwd(dir) # confirm the directory is set to the working directory...

# Import the CSV Data ----
hiccups = read_csv("./data/Hiccups.csv")
str(hiccups) # Peak into the Structure of the Data,,,