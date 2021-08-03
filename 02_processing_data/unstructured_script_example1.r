# - [Title] - Structured Data Sample - Compiling What We Have Learned So Far ----
### Note: The data includes text and classes so we could use this sample to address text classification...
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
# - [] - Bind Pre-process Text with Labels ----
merged.data = data.frame(cbind(data.text, data$class))
colnames(merged.data) = c("text", "class")
str(merged.data)
# - [] - Save the Processed Data ----
write.csv(merged.data,"data/merged.data.csv")