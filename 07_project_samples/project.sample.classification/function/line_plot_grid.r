line_plot_grid = function(x1,x2,x3,x4,x5){
  # - Line Plot ----
  print(paste0("Build Line Plot 1..."))
  p1 = ggplot(data, aes(x5, x1)) +
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
  print(paste0("Build Line Plot 2..."))
  p2 = ggplot(data, aes(x5, x2)) +
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
  print(paste0("Build Line Plot 3..."))
  p3 = ggplot(data, aes(x5, x3)) +
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
  p3
  print(paste0("Build Line Plot 4..."))
  p4 = ggplot(data, aes(x5, x4)) +
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
  p4
  print(paste0("Build Line Plot Grid..."))
  plot6 = plot_grid(p1, p2, p3, p4)
  assign("plot6",plot6, envir = globalenv())
  
}