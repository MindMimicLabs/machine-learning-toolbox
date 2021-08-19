# - Histogram in R ----
#install.packages('ggplot2')
library(ggplot2)
library(datasets)
data(iris)
data = iris 
p = ggplot(data, aes(Sepal.Width)) + 
    geom_histogram(binwidth = 0.4, color = 'green') + 
    xlab("X Variable") + 
    ylab("Frequency")
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