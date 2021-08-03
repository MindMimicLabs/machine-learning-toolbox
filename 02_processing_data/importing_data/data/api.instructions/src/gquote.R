

############### helper functions for google/ yahoo finance ###############
toTitle <- function (string){
  fn <- function(x){
    v <- unlist(strsplit(x, split = " "))
    u <- sapply(v, function(x){
      x <- tolower(x)
      substring(x, 1, 1) <- toupper(substring(x, 1, 1))
      x})
    paste(u, collapse = " ")
  }
  sapply(string, fn)
}

myUnlist <- function(data){
  x<-data[1]
  x<-x[[1]]
}

myEqualToParser <- function(l,str,numeric=TRUE, sep='='){
  x <- l[grep(str,l,ignore.case=TRUE)]

  if (length(x) == 0) {
    y <- NA
    return(y)
  }

  n <- nchar(x)
  y <- strsplit(x,sep)
  if (nchar(y) == n) {
    return(l)
  }

  y <- y[[1]]
  if ( numeric){
    return(as.numeric(y[2]))
  } else {
    return(y[2])
  }
}

myGetColumns <- function(l, str="COLUMNS", sep='='){
  x<-strsplit(l[grep(str,l,ignore.case=TRUE)],sep)
  x<-x[[1]]
  COLUMNS <- x[2]
  COLUMNS <- strsplit(COLUMNS,',')
  COLUMNS <- COLUMNS[[1]]
  return(COLUMNS)
}

myendpoints <- function(x,period){
  ## find number of days in data
  dates <- unique(as.Date(zoo::index(x)))
  n <- length(dates)
  last <- 0
  ind <- c(0)

  ## now loop through each day and process
  for (i in 1:n){
    xx <- zoo::index(x[paste(as.Date(dates[i]))])
    ind1 <- which((seq(1:length(xx))) %% period == 0)

    ## add final point if missing
    if (ind1[length(ind1)] < length(xx))
      ind1 <-c(ind1,length(xx))

    ind1 <- ind1 + last

    ind <- c(ind, ind1)
    last <- ind[length(ind)]
  }

  if (ind[1] == 0)
    ind<-ind[2:length(ind)]

  ind <- as.integer(ind)
  return(ind)
}

my.to.period <- function(x,period){
  freq<-unclass(xts::periodicity(x))
  if(freq$units != "mins")
    stop("unrecognized frequency unit")
  if (period <= round(freq$frequency,0))
    return(x)

  period <- round(period/freq$frequency,0)
  if (period ==1)
    return(x)

  cnames <- colnames(x)
  xx <- .Call("toPeriod", x, myendpoints(x,period), xts:::has.Vo(x),
              xts:::has.Vo(x, which = TRUE), xts:::has.Ad(x) && xts::is.OHLC(x),
              TRUE, cnames, PACKAGE = "xts")
  return(xx)
}

func_yahoo <- function(x){

  col1 <- as.POSIXct(as.numeric(x[1]) , origin = '1970-01-01')
  col2 <- as.numeric(x[2])
  col3 <- as.numeric(x[3])
  col4 <- as.numeric(x[4])
  col5 <- as.numeric(x[5])
  col6 <- as.numeric(x[6])

  return(c(col1,col2,col3,col4,col5,col6))
}

############### google/ yahoo finance functions #########################
getGoogleQuote <- function(ticker,period=1,interval=300,tz=NULL,auto.assign=TRUE){

  ticker<-strsplit(ticker,":")
  if (length(ticker[[1]]) != 1) {
    url <- paste("http://www.google.com/finance/getprices?i=",interval,"&p=",period,"d&f=d,o,h,l,c,v&df=cpct&q=",ticker[[1]][2],"&x=",ticker[[1]][1],sep="")
    ticker <- paste(ticker[[1]][1],".",ticker[[1]][2],sep="")

  } else {
    ticker <- ticker[[1]]
    url <- paste("http://www.google.com/finance/getprices?i=",interval,"&p=",period,"d&f=d,o,h,l,c,v&df=cpct&q=",ticker,sep="")
  }

  tmp <- RCurl::getURL(url)
  tmp <- strsplit(tmp,'\n')
  tmp <- tmp[[1]]

  ##exchange <- myEqualToParser(tmp,"EXCHANGE",FALSE,'%3D')
  ##market_open_minute <- myEqualToParser(tmp,"MARKET_OPEN_MINUTE")
  ##market_close_minute <- myEqualToParser(tmp,"MARKET_CLOSE_MINUTE")
  interval <- myEqualToParser(tmp,"INTERVAL")
  ##timezone_offset <- myEqualToParser(tmp,"TIMEZONE_OFFSET")
  columns <- toTitle(myGetColumns(tmp))
  columns <- sapply(columns, function(x,y){paste(ticker,".",x,sep="")})

  start <- grep("DATA",tmp) + 1
  tmp <- tmp[-c(1:start)]

  tmp <- strsplit(tmp,',')
  n <- length(tmp)
  col1<-rep(NA,n)
  col2<-rep(NA,n)
  col3<-rep(NA,n)
  col4<-rep(NA,n)
  col5<-rep(NA,n)
  col6<-rep(NA,n)
  col1<-as.POSIXct(col1,origin = '1970-01-01')

  for (i in 1:n){
    x<-myUnlist(tmp[i])

    if(substr(x[1],1,1) == "a"){
      tm = as.numeric(substr(x[1],2,nchar(x[1])))
      offset <- 0
    }else{
      offset <- as.numeric(x[1])*60
    }

    col1[i] <- as.POSIXct(tm + offset*interval/60, origin = '1970-01-01', tz=tz)

    col2[i] <- as.numeric(x[2])
    col3[i] <- as.numeric(x[3])
    col4[i] <- as.numeric(x[4])
    col5[i] <- as.numeric(x[5])
    col6[i] <- as.numeric(x[6])
  }

  if (sum(col3) == 0) col3 <- col2
  if (sum(col4) == 0) col4 <- col2
  if (sum(col5) == 0) col5 <- col2

  df <- data.frame(col2,col3,col4,col5,col6)
  colnames(df) <- columns[2:6]
  z<-xts::as.xts(df,col1)

  if(sum(z[,5])==0){
    z <- z[,-5]
  }

  if (auto.assign==FALSE)
    return(z)

  assign(ticker,z,1)
}

getYahooQuote <- function(ticker,period=1,interval=300,tz=NULL,auto.assign=TRUE){

  url<-paste("http://chartapi.finance.yahoo.com/instrument/1.0/",ticker,"/chartdata;type=quote;range=",period,"d/csv",sep="")

  ticker<-strsplit(ticker,"=")
  if (length(ticker[[1]]) != 1) {
    ticker <- paste(ticker[[1]][1],".",ticker[[1]][2],sep="")
  } else {
    ticker <- ticker[[1]]
  }

  tmp <- RCurl::getURL(url)
  tmp <- strsplit(tmp,'\n')
  tmp <- tmp[[1]]

  ##exchange <- myEqualToParser(tmp,"Exchange",FALSE,':')
  ##timezone_offset <- myEqualToParser(tmp,"gmtoffset",TRUE,':')/60
  columns <- toTitle(myGetColumns(tmp,"values",':'))
  ticker<-gsub("[^[:alnum:]]", "", ticker)
  columns <- sapply(columns, function(x,y){paste(ticker,".",x,sep="")})

  start <- grep("volume:",tmp)
  tmp <- tmp[-c(1:start)]

  tmp <- strsplit(tmp,',')

  x <- sapply(tmp, func_yahoo)
  x <- t(x)
  x <- data.frame(x)

  if (sum(x[,3]) == 0) x[,3] <- x[,2]
  if (sum(x[,4]) == 0) x[,4] <- x[,2]
  if (sum(x[,5]) == 0) x[,5] <- x[,2]

  x[,1] <- as.POSIXct(x[,1], origin = '1970-01-01', tz=tz)

  df <- data.frame(x[,5],x[,3],x[,4],x[,2],x[,6])
  colnames(df) <- c(columns[5],columns[3],columns[4],columns[2],columns[6])
  z<-xts::as.xts(df,x[,1])

  if(sum(z[,5])==0){
    z <- z[,-5]
  }

  z <- my.to.period(z,interval/60)

  if (auto.assign==FALSE)
    return(z)

  assign(ticker,z,1)
}

#'@title Download intraday data from Google Finance or Yahoo Finance
#'@description Download intraday data from Google Finance or Yahoo Finance in xts format
#'@param ticker ticker for the data to download, include source (exchange)
#'@param src source, can be either "google" or "yahoo"
#'@param period time period in integer # of days
#'@param interval time interval in integer # of minutes
#'@param tz timezone, set to NULL by default
#'@param auto.assign similar to getSymbols in quantmod, set to false by default
#'@param time.shift time shift to apply (in hours) to align the time stamp
#'@return an xts time series
#'@examples
#' x <- gquote::getIntradayPrice("INDEXSP:.INX",period=5, interval=5)
#' tail(x)
#'@export
getIntradayPrice<- function(ticker,src='google',period=1,interval=5,tz=NULL, auto.assign=FALSE, time.shift=0){

  if (src == 'google'){
    x <- getGoogleQuote(ticker,period,interval*60,tz,auto.assign)
  }
  else if(src == 'yahoo'){
    x <- getYahooQuote(ticker,period,interval*60,tz,auto.assign)
  }
  else {
    stop("illegal source")
  }
  zoo::index(x) <- zoo::index(x) + time.shift*60*60
  return(x)
}




