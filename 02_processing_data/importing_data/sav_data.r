#install.packages('haven')
library(haven) # load the package into the r enviroment...
dir = getwd() # locate and store the set working directory of the current r session
setwd(dir) # confirm the directory is set to the working directory...

# Import the SAV Data ----
chickflick = read_spss("./data/ChickFlick.sav")
str(chickflick) # Peak into the Structure of the Data,,,