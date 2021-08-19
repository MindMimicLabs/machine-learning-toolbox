# -[Title] - Sample Problem ----
# - [1] - Load Libraries ----
library(datasets)
library(psych)
library(Hmisc)
library(pastecs)
library(ggplot2)
library(GGally)
library(AppliedPredictiveModeling)
library(cowplot)
library(corrplot)
library(caret)
library(outlieR)
library(magrittr)
# - [2] - Import the Data ----
data(iris)
data = iris 
str(data)
# - [3] - Explore the Original Data ----
## - [a] - Summary Stat ----
summary(data)
stat.desc(data)
Hmisc::describe(iris)
psych::describe(iris)
## - [b] - Correlation ----
cor = cor(data[,c(1:4)]) # inputs must be in numeric data type...
print(cor)
png("pictures/cor.png")
print(corrplot::corrplot(cor))     # Plot 1 --> in the first page of PDF
dev.off() 
## - [c] - Visualize the Original Data ----
### - [] - Main Scatterplot 
p = ggplot(data, aes(Sepal.Width, Sepal.Length)) + 
  geom_jitter(aes(color = Species)) +
  geom_smooth(method = 'lm', color = 'black') +
  xlab('X Variable')+
  ylab('Y Variable')
p
cleanup = theme(panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                panel.background = element_blank(),
                axis.line.x = element_line(color = 'black'),
                axis.line.y = element_line(color = 'black'),
                legend.key = element_rect(fill = 'white'),
                text = element_text(size = 15))

p + cleanup
png("pictures/mainscatter.png")
print(p)     # Plot 1 --> in the first page of PDF
dev.off() 
# - optional scatterplot ----
t = with(data, qplot(data[,2], data[,4], colour=data[,ncol(data)], cex=0.2))
png("pictures/optional_scatters_1.png")
print(t)     # Plot 1 --> in the first page of PDF
dev.off() 
# - optional scatter plot matrix ----
g = ggpairs(data, title = 'Sample Iris Data')
png("pictures/scattermatrix.png")
print(g)     # Plot 1 --> in the first page of PDF
dev.off() 
# - optional scatterplot matrix ----
transparentTheme(trans = .4)
tt = featurePlot(x = data[, 1:4], 
            y = data$Species, 
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))
png("pictures/optional_scatter_matrix.png")
print(tt)     # Plot 1 --> in the first page of PDF
dev.off() 
### - [] - Histogram ----
# - HSW
p1 = ggplot(data, aes(Sepal.Width,color = Species)) + 
  geom_histogram(binwidth = 0.4) + 
  xlab("Sepal.Width") + 
  ylab("Frequency")
p1
# - HPW
p2 = ggplot(data, aes(Petal.Width,color = Species)) + 
  geom_histogram(binwidth = 0.4) + 
  xlab("Petal.Width") + 
  ylab("Frequency")
p2
# - HSL
p3 = ggplot(data, aes(Sepal.Length,color = Species)) + 
  geom_histogram(binwidth = 0.4) + 
  xlab("Petal.Width") + 
  ylab("Frequency")
p3
# - HPL
p4 = ggplot(data, aes(Petal.Length,color = Species)) + 
  geom_histogram(binwidth = 0.4) + 
  xlab("Petal.Width") + 
  ylab("Frequency")
p4
plot_grid(p1,p2,p3,p4)

# - Barplot ---- 

p1 = ggplot(data, aes(Species, Sepal.Length)) +
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
p1
p2 = ggplot(data, aes(Species, Sepal.Width)) +
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
p2
p3 = ggplot(data, aes(Species, Petal.Length)) +
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
p3
p4 = ggplot(data, aes(Species, Petal.Width)) +
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
p4
plot_grid(p1,p2,p3,p4)

# - Boxplot ----
featurePlot(x = data[, 1:4], 
            y = data$Species, 
            plot = "box", 
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(4,1 ), 
            auto.key = list(columns = 2))

# - Density plot ----
transparentTheme(trans = .9)
featurePlot(x = data[, 1:4], 
            y = data$Species,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(4, 1), 
            auto.key = list(columns = 3))

# - Line Plot ----
p1 = ggplot(data, aes(Species, Sepal.Length)) +
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
p1
p2 = ggplot(data, aes(Species, Sepal.Width)) +
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
p2
plot_grid(p1,p2)

# - [] - Preprocessing ---- 
out = data[,1:4]
out = out %>% outlieR::impute(flag = NULL, fill = "mean", 
                                level = 0.1, nmax = NULL,
                                side = NULL, crit = "lof", 
                                k = 5, metric = "euclidean", q = 3)
data = data.frame(out,Species = data[,5])
preProClean <- preProcess(x = data, method = c("scale", "center"))
data <- predict(preProClean, data %>% na.omit)
print(str(data))
## - [c] - Visualize the Processed Data ----
### - [] - Main Scatterplot 
p = ggplot(data, aes(Sepal.Width, Sepal.Length)) + 
  geom_jitter(aes(color = Species)) +
  geom_smooth(method = 'lm', color = 'black') +
  xlab('X Variable')+
  ylab('Y Variable')
p
cleanup = theme(panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                panel.background = element_blank(),
                axis.line.x = element_line(color = 'black'),
                axis.line.y = element_line(color = 'black'),
                legend.key = element_rect(fill = 'white'),
                text = element_text(size = 15))

p + cleanup

# - optional scatterplot ----
with(data, qplot(data[,2], data[,4], colour=data[,ncol(data)], cex=0.2))

# - optional scatter plot matrix ----
ggpairs(data, title = 'Sample Iris Data')

# - optional scatterplot matrix ----
transparentTheme(trans = .4)
featurePlot(x = data[, 1:4], 
            y = data$Species, 
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))
### - [] - Histogram ----
# - HSW
p1 = ggplot(data, aes(Sepal.Width,color = Species)) + 
  geom_histogram(binwidth = 0.4) + 
  xlab("Sepal.Width") + 
  ylab("Frequency")
p1
# - HPW
p2 = ggplot(data, aes(Petal.Width,color = Species)) + 
  geom_histogram(binwidth = 0.4) + 
  xlab("Petal.Width") + 
  ylab("Frequency")
p2
# - HSL
p3 = ggplot(data, aes(Sepal.Length,color = Species)) + 
  geom_histogram(binwidth = 0.4) + 
  xlab("Petal.Width") + 
  ylab("Frequency")
p3
# - HPL
p4 = ggplot(data, aes(Petal.Length,color = Species)) + 
  geom_histogram(binwidth = 0.4) + 
  xlab("Petal.Width") + 
  ylab("Frequency")
p4
plot_grid(p1,p2,p3,p4)

# - Barplot ---- 

p1 = ggplot(data, aes(Species, Sepal.Length)) +
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
p1
p2 = ggplot(data, aes(Species, Sepal.Width)) +
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
p2
p3 = ggplot(data, aes(Species, Petal.Length)) +
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
p3
p4 = ggplot(data, aes(Species, Petal.Width)) +
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
p4
plot_grid(p1,p2,p3,p4)

# - Boxplot ----
featurePlot(x = data[, 1:4], 
            y = data$Species, 
            plot = "box", 
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(4,1 ), 
            auto.key = list(columns = 2))

# - Density plot ----
transparentTheme(trans = .9)
featurePlot(x = data[, 1:4], 
            y = data$Species,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(4, 1), 
            auto.key = list(columns = 3))

# - Line Plot ----
p1 = ggplot(data, aes(Species, Sepal.Length)) +
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
p1
p2 = ggplot(data, aes(Species, Sepal.Width)) +
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
p2

#pdf("grid.pdf")
#print(plot_grid(p1,p2))     # Plot 1 --> in the first page of PDF
#dev.off() 

png("grid.png")
print(plot_grid(p1,p2))     # Plot 1 --> in the first page of PDF
dev.off() 