# - Research Classifier ----
setwd("C:/Users/jonat/Desktop/project.sample.classification")
# - Load the libraries ----
source("scripts/libs.r")
# - Import and Explore the Original Data ----
source("scripts/import_data.r")
# - Set Variables Original ----
x1 = data$Sepal.Length
x2 = data$Sepal.Width
x3 = data$Petal.Length
x4 = data$Petal.Width
x5 = data$Species
d1 = data[,1:4]
# - Explore the Original ----
source("scripts/explore.O.r")

# - Preprocess and Explore the Prep Data ----
source("scripts/prep.r")
# - Import the Prep Data ----
source("scripts/import_prep.r")
# - Set Variables Prep ----
x1 = prep$Sepal.Length
x2 = prep$Sepal.Width
x3 = prep$Petal.Length
x4 = prep$Petal.Width
x5 = prep$x5
d1 = prep[,1:4]
# - Explore the Prep ----
source("scripts/explore.P.r")

# - Classification Modeling ----
source("modeling/modeling.r")
