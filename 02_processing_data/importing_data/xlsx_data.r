#install.packages('readxl')
library(readxl) # load the package into the r enviroment...
dir = getwd() # locate and store the set working directory of the current r session
setwd(dir) # confirm the directory is set to the working directory...

# Import the XLSX Data ----
Texting <- read_excel("./data/Texting.xlsx")
str(Texting) # Peak into the Structure of the Data,,,