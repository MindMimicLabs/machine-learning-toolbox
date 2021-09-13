bar_plot_grid = function(x1,x2,x3,x4,x5){
  # - Barplot ---- 
  print(paste0("Build Barplot 1..."))
  p1 = ggplot(data, aes(x5, x1)) +
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
  print(paste0("Build Barplot 2..."))
  p2 = ggplot(data, aes(x5, x2)) +
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
  print(paste0("Build Barplot 3..."))
  p3 = ggplot(data, aes(x5, x3)) +
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
  print(paste0("Build Barplot 4..."))
  p4 = ggplot(data, aes(x5, x4)) +
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
  print(paste0("Build Barplot Grid..."))
  plot3 = plot_grid(p1,p2,p3,p4)
  assign("plot3",plot1, envir = globalenv())
}