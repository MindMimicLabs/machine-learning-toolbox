# - [0] - Import Libraries ----
library(stringr)
library(R.utils)
library(readtext)
library(data.table)
library(tm)
library(stringr)
library(udpipe)

# - [1] - SetWd ----
setwd("C:/Users/jonat/Desktop/project.sub.ner")

# - [2] - Import Subtitle Data ---- 
#save and store a gzipped text file from http://opus.nlpl.eu/OpenSubtitles.php
#make sure to download the datafile from the second table under 'Statistics and TMX/Moses Downloads'
list_of_txt <- list.files(path = "data/", recursive = TRUE,
                          pattern = "\\.gz$", 
                          full.names = TRUE) #capture list of gzipped files...

for(i in seq_along(length(list_of_txt))) {
  text = lapply(list_of_txt, gunzip) #import gzipped files and store as unzipped text files...
}
for(i in seq_along(length(text))) {
  text.list = lapply(text, readtext) #read text files into a list of dataframes...
  text.list = lapply(text.list, data.frame)
}
for(i in seq_along(length(text.list))) {
  string = rbindlist(text.list) #row bind list of dataframes into a dataframe....
}

# - [3] - Pre-process the texts ----
data.text = string$text
data.text = tolower(data.text)
data.text = iconv(data.text, "latin1", "ASCII", sub = " ")
data.text = tm::stripWhitespace(data.text)

# - [4] - Annotation of the texts ----
dl = udpipe_download_model(language = "afrikaans-afribooms") #download the language model required...
annotate = udpipe(data.text, "afrikaans-afribooms") #annotate the texts...
ann.df = as.data.frame(annotate) #store as a dataframe...
str(ann.df) #structure of dataframe...

# - [5] - Reshaping the dataframe ----
#required libraries...
library(plyr)
library(dplyr)
library(data.table)

# Subset DF to only target variables...
subset.df = ann.df[, c('sentence_id','token','upos')]

# Loop through to concatenate token + upos into unique sentences...
p <- function(v) {
  Reduce(f=paste0, x = v)
}

for(i in seq_along(length(subset.df$sentence_id))) {
  sub.lab.tok = (c(paste(subset.df$token, subset.df$upos))) #combine tokens and upos...
  mod.df = data.frame(subset.df, sub.lab.tok) #convert into a df...
  colnames(mod.df) = c("sentence_id","token", "upos", "upos.labels") #relabel columns of df...
  mod.df$upos.labels = as.character(mod.df$upos.labels) #convert token_upos to characters...
  mod.df1 = aggregate(upos.labels~sentence_id, data = mod.df, paste0, collapse=" ") #aggregate df
  mod.df2 = merge(mod.df, mod.df1, by = "sentence_id", all = T) #merge mod.df and mod.df1 into 1 df... 
  colnames(mod.df2) = c("sentence_id", "token", "upos", "token_upos", "sent_upos") #relabel columns of df...
}
str(mod.df2)


#frequency count of unique 'token_upos' for only [VBD< NOUN, ADJ]...
#subset the dataframe to only include tagged sentences including the target POS...
freq.verb = str_count(mod.df2$sent_upos, "VERB") #count of VERBs in each string...
freq.noun = str_count(mod.df2$sent_upos, "NOUN") #count of each NOUN in each string...
freq.adj = str_count(mod.df2$sent_upos, "ADJ") #count of each ADJ in each string...
freq.df = data.frame(cbind(mod.df2, freq.verb, freq.noun, freq.adj)) #combined into a single df...
str(freq.df)

# - [6] - Save the final dataframe as a csv ----
fq = 1 #change condition of frequency for POS targets...
final.df = filter(freq.df, (freq.df$freq.verb > fq & freq.df$freq.noun > fq & freq.df$freq.adj > fq)) #filter based sentences containing POS Tags > fq...
str(final.df)

write.csv(final.df,"csv/final.df.csv", row.names = FALSE)