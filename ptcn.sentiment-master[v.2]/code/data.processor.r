# - [0] - Load Required Packages ----

library(keras)
library(tensorflow)
library(xml2)
library(dplyr)
library(plyr)
library(searcher)
library(tokenizers)
library(tm)
library(stringr)
library(topicmodels)
library(doParallel)
library(ggplot2)
library(scales)
library(qdapDictionaries)
library(data.table)
library(readtext)
library(hunspell)
library(ggplot2)
library(qdap)
library(sentimentr)
library(readr)

## [1] - Importing Data...

#list of text
list_of_txt <- list.files(path = "data/.", recursive = TRUE,
                          pattern = "\\.txt$", 
                          full.names = TRUE)

for(i in seq_along(length(list_of_txt))) {
  text = lapply(list_of_txt, readtext::readtext)
}


#list of tables
list_of_csv <- list.files(path = "data/.", recursive = TRUE,
                          pattern = "\\.csv$", 
                          full.names = TRUE)


for(i in seq_along(length(list_of_csv))) {
  csv= lapply (list_of_csv, read.csv, header=FALSE, sep=",")
}




for(i in seq_along(length(list_of_csv))) {
  csv[[i]][[4]] = gsub("^Not Voting", "NV",  csv[[i]][[4]])
}


#list of bill texts votes (classes)
for(i in seq_along(list_of_csv)) {
  yea[i] = c(sum(csv[[i]][[4]] == "Yea"))
  nay[i] = c(sum(csv[[i]][[4]] == "Nay"))  
  sums = cbind(yea,nay)
  class[i] = ifelse(sums[i,1] > sums[i,2], 1, 0)
}

#dataframe of text 
for(i in seq_along(length(text))) {
  string = rbindlist(text)
}


data.final = cbind(string[,2], class[1:3668])
colnames(data.final) = c("text", "class")


# Create an equal number of 1 and 0 classes in the dataset. 
#Splitting Subsets to 
split_0 = data.final[which(data.final$class == 0), ]
split_1 = data.final[which(data.final$class == 1), ]

table(split_0$class)
table(split_1$class)

data.even = rbind(split_0[1:711,], split_1[1:711,])
N4 =  nrow(data.even)
ind4 = sample(N4, N4*1, replace = FALSE) 
data4  = data.even[ind4,]
apply(data4,2,function(x) sum(is.na(x)))#Check for Missing...
str(data4)

## [2] - Data Pre-processing...
text <- data4$text
text = tolower(text)
text = iconv(text, "latin1", "ASCII", sub = " ")
text = gsub("^NA| NA ", " ", text)
text = tm::removeWords(text, stopwords(kind = "SMART"))
text = tm::removePunctuation(text)
text = tm::removeNumbers(text)
text = tm::stripWhitespace(text)
text = tm::stemDocument(text)

data4 = data.frame(cbind(text, data4$class))
str(data4)

write.csv(data4, "data.csv")
