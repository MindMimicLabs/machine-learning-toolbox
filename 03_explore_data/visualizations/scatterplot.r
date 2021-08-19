# - Scattterplot in R ----
#install.packages('ggplot2')
library(ggplot2)
library(datasets)
data(iris)
data = iris 
p = ggplot(data, aes(Sepal.Width, Sepal.Length)) + 
  geom_point() +
  geom_smooth(method = 'lm', color = 'black') +
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

# Grouped Scatter Plot ----
Species = as.factor(data$Species)
p = ggplot(data, aes(Sepal.Width, Sepal.Length)) + 
  geom_point(aes(colour = Species)) +
  geom_smooth(method = 'lm', color = 'black') +
  xlab('X Variable') +
  ylab('Y Variable') 
  cleanup
  p
  
  # Optional Method ----
  with(data, qplot(data[,1], data[,2], colour=data[,ncol(data)], cex=0.2))