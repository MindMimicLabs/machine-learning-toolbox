### - Summary Stats ----
print(summary(data))
### - Main Scatterplot ----
source("function/scatter_plot_grid.r")
scatter_plot_grid(x1,x2,x3,x4,x5)
png("pictures/mainscatter_O.png")
print(plot1)     # Plot 1 --> in the first page of PDF
dev.off() 
### - [] - Histogram ----
source("function/hist_plot_grid.r")
hist_plot_grid(x1,x2,x3,x4,x5)
png("pictures/hist_O.png")
print(plot2)     # Plot 1 --> in the first page of PDF
dev.off()
# - Barplot ---- 
source("function/bar_plot_grid.r")
bar_plot_grid(x1,x2,x3,x4,x5)
png("pictures/bar_O.png")
print(plot3)     # Plot 1 --> in the first page of PDF
dev.off()
# - Line Plot ----
source("function/line_plot_grid.r")
line_plot_grid(x1,x2,x3,x4,x5)
png("pictures/line_O.png")
print(plot6)     # Plot 1 --> in the first page of PDF
dev.off()
# - Correlation Plot ----
print(paste0("Build Correlation Plot..."))
cor = cor(d1) # inputs must be in numeric data type...
png("pictures/cor_O.png")
print(corrplot::corrplot(cor))    
dev.off()