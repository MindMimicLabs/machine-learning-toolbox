# Process Text ----
# install.packages('tm')
library(tm)
# install.packages('stringr')
library(stringr)
data <- data.frame(read.csv("./data/text.csv"))
data = tail(data,100)
colnames(data) = c('text','class')
str(data)
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

# merge text w/ labels again...
merged.data = data.frame(cbind(data.text, data$class))
colnames(merged.data) = c("text", "class")
head(merged.data)

write.csv(merged.data, "merged.data.csv")



# - Tokenizing Text ----
if (!require('keras')) {
  install.packages("keras", dependencies = TRUE)
  library(keras)
}
tokenizer <- text_tokenizer(num_words = 4582)
tokenizer %>% fit_text_tokenizer(data.text)

# Convert Text to Sequences ----
text_seqs <- texts_to_sequences(tokenizer, data.text)
print(text_seqs[1])

# Pad Sequences ----
# same length all inputs...
pad <- text_seqs %>% pad_sequences(maxlen = 200)
print(head(pad))
str(pad)

# Document Term Matrix Object ----
data.dtm = tm::Corpus(VectorSource(data.text))
dtm <- tm::DocumentTermMatrix(data.dtm,control = list(freq = function(x) termFreq(x,control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))))
rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document
dtm   <- dtm[rowTotals > 0, ]     #remove all docs without words
inspect(dtm)
