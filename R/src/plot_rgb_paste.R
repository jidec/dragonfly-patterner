plotRGBColor <- function(rgb,max=255) {
  # todo - finish allowing matrix of multiple rgbs
  #rgb <- rbind(rgb, rgb2)
  #rgb[,1], rgb[,2]
  col <- rgb(rgb[1], rgb[2], rgb[3], maxColorValue = max)
  return(plot(c(1), col = col, pch = 15, cex = 40, axes = FALSE, ylab='',xlab=''))
}

plotRGBColors <- function(rgbs,max=255){
  for(r in 1:nrow(rgbs)){
    plotRGBColor(rgbs[r,],max=max)
  }
}