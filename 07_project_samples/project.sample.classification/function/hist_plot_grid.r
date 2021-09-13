hist_plot_grid = function(x1,x2,x3,x4,x5){
  ### - [] - Histogram ----
  # - HSW
  print(paste0("Build Histogram 1..."))
  p1 = ggplot(data, aes(x1,color = x5)) + 
    geom_histogram(binwidth = 0.4) + 
    xlab("x1") + 
    ylab("Frequency")
  p1
  # - HPW
  print(paste0("Build Histogram 2..."))
  p2 = ggplot(data, aes(x2,color = x5)) + 
    geom_histogram(binwidth = 0.4) + 
    xlab("x2") + 
    ylab("Frequency")
  p2
  # - HSL
  print(paste0("Build Histogram 3..."))
  p3 = ggplot(data, aes(x3,color = x5)) + 
    geom_histogram(binwidth = 0.4) + 
    xlab("x3") + 
    ylab("Frequency")
  p3
  # - HPL
  print(paste0("Build Histogram 4..."))
  p4 = ggplot(data, aes(x4,color = x5)) + 
    geom_histogram(binwidth = 0.4) + 
    xlab("x4") + 
    ylab("Frequency")
  p4
  print(paste0("Build Histogram Grid..."))
  plot2 = plot_grid(p1,p2,p3,p4)
  assign("plot2",plot2, envir = globalenv())
}