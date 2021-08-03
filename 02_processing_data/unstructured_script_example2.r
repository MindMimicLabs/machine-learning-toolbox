# - [Title] - Structured Data Sample - Compiling What We Have Learned So Far ----
### Note: Just preprocess the text and feed into the DTM chunk of code for Topics Modeling...
# - [] - Import Libraries ----
library(readr)
library(tm)
library(stringr)
# - [] - Import a UnStructured Data Table ----
data <- data.frame(read.csv("./data/text.csv"))
data = tail(data,100)
colnames(data) = c('text','class')
str(data)
# - [] - Pre-process the text ----
text = c(lapply(data$text, as.character)) # convert to string/character format...
data.text = text
data.text = tolower(data.text) # convert letter casing to lower...
data.text = tm::removeWords(data.text, stopwords("SMART")) # remove words like t(the, a, an, etc...)
data.text = iconv(data.text, "latin1", "ASCII", sub = " ") # remove special characters...
data.text = gsub("^NA| NA ", " ", data.text) # remove all NA values/patterns...
data.text = gsub("^br| br ", " ", data.text)
data.text = tm::removePunctuation(data.text) # remove punctuation...
data.text = tm::removeNumbers(data.text) # remove numbers...
data.text = tm::stripWhitespace(data.text) # remove whitespace...
# - [] - Convert Text into DTM ----
data.dtm = tm::Corpus(VectorSource(data.text))
dtm <- tm::DocumentTermMatrix(data.dtm,control = list(freq = function(x) termFreq(x,control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))))
rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document
dtm   <- dtm[rowTotals > 0, ]     #remove all docs without words
inspect(dtm)