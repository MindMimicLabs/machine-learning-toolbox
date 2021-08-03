# Quantmod API for S&P500 Stock Data *(Daily/Intraday)*
if (!require("quantmod")) {
  install.packages("quantmod")
  library(quantmod)
}
if (!require("remotes")) {
  install.packages("remotes")
  library(remotes)
}
source("data/api.instructions/src/gquote.R")
# Daily Request ----
start <- as.Date(Sys.Date()-(365*5)) # start date. 
end <- as.Date(Sys.Date()) # current date. 
getSymbols("GOOG", src = "yahoo", from = start, to = end) # Feed symbol from s&p500. 
data = GOOG # call xts object into env. 
colnames(data) = c("Open", "High", "Low", "Close", "Volume", "Adjusted") # set column names in xts object. 
str(data)

# Period Based Request ----
setDefaults(getSymbols.av, api.key="4AIIJTFH81GJTWWC")
#api.key = c("4AIIJTFH81GJTWWC") 
# For an api go to https://www.alphavantage.co/.
# ...click on "Get Your Free API Key Today" button.
# ... on the next screen that populates, fill in the application. 
# ... you require an email for the application. 
# ... make sure to select the student status for the api key. 
# ... copy and store the api key that generates on the screen after submitting the application. 
getSymbols.av("IBM", 
              env = parent.frame(),
              src="av",
              #api.key = api.key,
              return.class = "xts",
              periodicity = "daily", # one of "daily", "weekly", "monthly", or "intraday"
              adjusted = FALSE,
              interval = "1min", # one of "1min", "5min", "15min", "30min", or "60min" (intraday data only)
              output.size = "compact", # either "compact" or "full
              data.type = "json") # either "json" or "csv"
data = IBM
str(data)
# Intraday Request ----
n = 1 # number of days.
getIntradayPrice("MMM", src = "google", period = n, interval = 5,
                 tz = NULL, auto.assign = FALSE, time.shift = 0)
data2 = MMM
str(data2)
summary(data2)
head(data2)
