# attempt to subtractively mix colors (should prolly really do so by converting to spec and using WGM)
getColorAvg <- function(rgb_colors_list, percents){
  rgb <- c()
  for(c in 1:length(rgb_colors_list)){
    color <- rgb_colors_list[[c]]
    rgb <- c(rgb,color * percents[[i]])
  }
  rgb <- rgb / length(rgb_colors_list)
  return(rgb)
}
