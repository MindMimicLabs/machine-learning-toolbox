setwd("C:/Users/jonat/Desktop/cu-hsp-learning-main/04_prepare.tools")
# - Install R Packages ----
if (!require('keras')) {
  install.packages("keras", dependencies = TRUE)
  library(keras)
}
#install_keras()
if (!require('tensorflow')) {
  install.packages("tensorflow", dependencies = TRUE)
  library(tensorflow)
}
#install_tensorflow(version = "1.15.0")
if (!require('tfruns')) {
  install.packages("tfruns", dependencies = TRUE)
  library(tfruns)
}
if (!require('xml2')) {
  install.packages("xml2", dependencies = TRUE)
  library(xml2)
}
if (!require('dplyr')) {
  install.packages("dplyr", dependencies = TRUE)
  library(dplyr)
}
if (!require('plyr')) {
  install.packages("plyr", dependencies = TRUE)
  library(plyr)
}
if (!require('searcher')) {
  install.packages("searcher", dependencies = TRUE)
  library(searcher)
}
if (!require('tokenizers')) {
  install.packages("tokenizers", dependencies = TRUE)
  library(tokenizers)
}
if (!require('tm')) {
  install.packages("tm")
  library(tm)
}
if (!require('stringr')) {
  install.packages("stringr", dependencies = TRUE)
  library(stringr)
}
if (!require('topicmodels')) {
  install.packages("topicmodels", dependencies = TRUE)
  library(topicmodels)
}
if (!require('doParallel')) {
  install.packages("doParallel", dependencies = TRUE)
  library(doParallel)
}
if (!require('colorspace')) {
  install.packages("colorspace", dependencies = TRUE)
  library(colorspace)
}
if (!require('ggplot2')) {
  install.packages("ggplot2", dependencies = TRUE)
  library(ggplot2)
}
if (!require('scales')) {
  install.packages("scales", dependencies = TRUE)
  library(scales)
}
if (!require('qdapDictionaries')) {
  install.packages("qdapDictionaries", dependencies = TRUE)
  library(qdapDictionaries)
}
if (!require('data.table')) {
  install.packages("data.table", dependencies = TRUE)
  library(data.table)
}
if (!require('readtext')) {
  install.packages("readtext", dependencies = TRUE)
  library(readtext)
}
if (!require('hunspell')) {
  install.packages("hunspell", dependencies = TRUE)
  library(hunspell)
}
Sys.setenv(JAVA_HOME="C:/Program Files/Java/jre1.8.0_281/")
if (!require('rJava')) {
  install.packages('./packages/rJava_0.9-11.tar.gz', repos = NULL, type ='source')
  library(rJava)
}
if (!require('devtools')) {
  install.packages("devtools", dependencies = TRUE)
  library(devtools)
}
if (!require('igraph')) {
  install.packages("igraph", dependencies = TRUE)
  library(igraph)
}
if (!require('qdap')) {
  install.packages('./packages/qdap_2.3.2.tar.gz', repos = NULL, type ='source')
  library(qdap)
}
if (!require('sentimentr')) {
  install.packages("sentimentr", dependencies = TRUE)
  library(sentimentr)
}
if (!require('readr')) {
  install.packages("readr", dependencies = TRUE)
  library(readr)
}
if (!require('readxl')) {
  install.packages("readxl", dependencies = TRUE)
  library(readxl)
}
if (!require('caret')) {
  install.packages("caret", dependencies = TRUE)
  library(caret)
}
if (!require('openNLP')) {
  install.packages('./packages/openNLP_0.2-6.tar.gz', repos = NULL, type ='source')
  library(openNLP)
}
if (!require('openNLPmodels.en')) {
  install.packages("openNLPmodels.en", dependencies=TRUE, repos = "http://datacube.wu.ac.at/")
  library(openNLPmodels.en)
}
if (!require('gsubfn')) {
  install.packages("gsubfn")
  library(gsubfn)
}
if (!require("purrr")) {
  install.packages("purrr")
  library(purrr)
}
if (!require("quantmod")) {
  install.packages("quantmod")
  library(quantmod)
}  
if (!require("TTR")) {
  install.packages("TTR")
  library(TTR)
}
if (!require("xts")) {
  install.packages("xts")
  library(xts)
}
if (!require("data.table")) {
  install.packages("data.table")
  library(data.table)
}
if (!require("dummies")) {
  install.packages("dummies")
  library(dummies)
}
if (!require("caret")) {
  install.packages("caret")
  library(caret)
}
if (!require("caretEnsemble")) {
  install.packages("caretEnsemble")
  library(caretEnsemble)
}
if (!require("lubridate")) {
  install.packages("lubridate")
  library(lubridate)
}
if (!require("magrittr")) {
  install.packages("magrittr")
  library(magrittr)
}
if (!require("rowr")) {
  install.packages('./packages/rowr_1.1.3.tar.gz', repos = NULL, type ='source')
  library(rowr)
}
if (!require("anytime")) {
  install.packages("anytime")
  library(anytime)
}
if (!require("foreach")) {
  install.packages("foreach")
  library(foreach)
}
if (!require("e1071")) {
  install.packages("e1071")
  library(e1071)
}
if (!require("PerformanceAnalytics")) {
  install.packages("PerformanceAnalytics")
  library(PerformanceAnalytics)
}
if (!require("discretization")) {
  install.packages("discretization")
  library(discretization)
}
if (!require("pipeR")) {
  install.packages("pipeR")
  library(pipeR)
}
if (!require("mice")) {
  install.packages("mice")
  library(mice)
}
if (!require("remotes")) {
  install.packages("remotes")
  library(remotes)
}
if (!require("outlieR")) {
  remotes::install_github("rushkin/outlieR")
  library(outlieR)
}
if (!require("hablar")) {
  install.packages("hablar")
  library(hablar)
}
if (!require("NoiseFiltersR")) {
  install.packages("NoiseFiltersR")
  library(NoiseFiltersR)
}
if (!require("parallel")) {
  install.packages("parallel")
  library(parallel)
}
if (!require("kernlab")) {
  install.packages("kernlab")
  library(kernlab)
}
if (!require("nnet")) {
  install.packages("nnet")
  library(nnet)
}
if (!require("glmnet")) {
  install.packages('./packages/glmnet_2.0-16.tar.gz', repos = NULL, type ='source')
  library(glmnet)
}
if (!require("Matrix")) {
  install.packages("Matrix")
  library(Matrix)
}
if (!require("MASS")) {
  install.packages("MASS")
  library(MASS)
}
if (!require("C50")) {
  install.packages("C50")
  library(C50)
}
if (!require("naivebayes")) {
  install.packages("naivebayes")
  library(naivebayes)
}
if (!require("ncar")) {
  install.packages("ncar")
  library(ncar)
}
if (!require("bsts")) {
  install.packages('./packages/bsts_0.8.0.tar.gz', repos = NULL, type ='source')
  library(bsts)
}
if (!require("plumber")) {
  install.packages("plumber")
  library(plumber)
}
if (!require("arm")) {
  install.packages("arm")
  library(arm)
}
if (!require('pastecs')) {
  install.packages("pastecs", dependencies = TRUE)
  library(pastecs)
}
if (!require('Hmisc')) {
  install.packages("Hmisc", dependencies = TRUE)
  library(Hmisc)
}
if (!require('psych')) {
  install.packages("psych", dependencies = TRUE)
  library(psych)
}
if (!require('AppliedPredictiveModeling')) {
  install.packages("AppliedPredictiveModeling", dependencies = TRUE)
  library(AppliedPredictiveModeling)
}
if (!require('viridis')) {
  install.packages("viridis", dependencies = TRUE)
  library(viridis)
}
if (!require('timetk')) {
  install.packages("timetk", dependencies = TRUE)
  library(timetk)
}
if (!require('NbClust')) {
  install.packages("NbClust", dependencies = TRUE)
  library(NbClust)
}
if (!require("magrittr")) {
  install.packages("magrittr")
  library(magrittr)
}
if (!require("reticulate")) {
  install.packages('./packages/reticulate-1.15.tar.gz', repos = NULL, type ='source')
  library(reticulate)
}
if (!require("dict ")) {
  if (!require("devtools")) install.packages("devtools")
  devtools::install_github("mkuhn/dict")
}
