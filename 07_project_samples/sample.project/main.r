# Main Script ---- 
setwd("C:/Users/jonat/Desktop/sample.project")
library(logr)
log_open()
source("scripts/library.r")
source("scripts/import_data.r")
source("scripts/preprocess.r")
source("scripts/model/dt.r")
source("scripts/model/rf.r")
log_close()
