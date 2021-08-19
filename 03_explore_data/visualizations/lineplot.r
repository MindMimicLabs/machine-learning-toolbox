# Barplots in R ----
#install.packages('ggplot2')
library(ggplot2)
library(datasets)
data(iris)
data = iris 
p = ggplot(data, aes(Species, Sepal.Length)) +
  stat_summary(fun.y = mean, ##adds the points
               geom = "point") +
  stat_summary(fun.y = mean, ##adds the line
               geom = "line",
               aes(group=1)) +
  stat_summary(fun.data = mean_cl_normal, ##adds the error bars
               geom = "errorbar", 
               width = .2) +
  xlab('X Variable')+
  ylab('Y Variable')
p

# Optional Clean Up Code ----
cleanup = theme(panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                panel.background = element_blank(),
                axis.line.x = element_line(color = 'black'),
                axis.line.y = element_line(color = 'black'),
                legend.key = element_rect(fill = 'white'),
                text = element_text(size = 15))

p + cleanup

# Optional Method ----
library(quantmod)
start <- as.Date(Sys.Date()-(365*5))
end <- as.Date(Sys.Date())
getSymbols("GOOG", src = "yahoo", from = start, to = end)
data = GOOG
plot(data[, "Close"], main = "Close Price") #Close Price...
colnames(data) = c("Open", "High", "Low", "Close", "Volume", "Adjusted")