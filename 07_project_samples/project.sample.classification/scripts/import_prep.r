# - Import the Data ----
library(readr)
prep <- read_csv("data/prep.csv")
prep <- data.frame(prep[,-1])
print(str(prep))
print(head(prep))