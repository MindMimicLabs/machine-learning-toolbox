scatter_plot_grid = function(x1,x2,x3,x4,x5){
  print(paste0("Build Scatterplot 1..."))
  p = ggplot(data, aes(x1, x2)) + 
    geom_jitter(aes(color = x5)) +
    geom_smooth(method = 'lm', color = 'black') +
    xlab('X Variable = x1')+
    ylab('Y Variable = x2')
  p
  cleanup = theme(panel.grid.major = element_blank(),
                  panel.grid.minor = element_blank(),
                  panel.background = element_blank(),
                  axis.line.x = element_line(color = 'black'),
                  axis.line.y = element_line(color = 'black'),
                  legend.key = element_rect(fill = 'white'),
                  text = element_text(size = 15))
  
  p1 = p + cleanup
  print(paste0("Build Scatterplot 2..."))
  p2 = ggplot(data, aes(x3, x4)) + 
    geom_jitter(aes(color = x5)) +
    geom_smooth(method = 'lm', color = 'black') +
    xlab('X Variable = x3')+
    ylab('Y Variable = x4')
  p2
  cleanup = theme(panel.grid.major = element_blank(),
                  panel.grid.minor = element_blank(),
                  panel.background = element_blank(),
                  axis.line.x = element_line(color = 'black'),
                  axis.line.y = element_line(color = 'black'),
                  legend.key = element_rect(fill = 'white'),
                  text = element_text(size = 15))
  
  p2 = p + cleanup
  print(paste0("Build Scatterplot Grid..."))
  plot1 = plot_grid(p1,p2)
  assign("plot1",plot1, envir = globalenv())
}