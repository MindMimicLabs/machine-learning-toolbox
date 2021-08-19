# Barplots in R ----
#install.packages('ggplot2')
library(ggplot2)
library(datasets)
data(iris)
data = iris 
p = ggplot(data, aes(Species, Sepal.Length)) +
  stat_summary(fun.y = mean,
               geom = "bar",
               fill = "White", 
               color = "Black") + 
  stat_summary(fun.data = mean_cl_normal, 
               geom = "errorbar", 
               position = position_dodge(width = 0.90), 
               width = 0.2) + 
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