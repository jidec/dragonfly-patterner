
library(recolorize)

dir <- "D:/wing-color/data/segments"
write_dir <- "D:/wing-color/data/patterns"

# define a custom recolorize function
recolorizeWingSegmentDir <- function(dir,write_dir,start_index=1) {
  
  wings <- list.files(path=dir)#,full.names=TRUE)
  wings <- wings[start_index:length(wings)]
  counter <- 0
  
  #w <- "MLM-000009.tif_fore.png"
  #w <- wings[7]
  for(w in wings){
    # blur image:
    rc0 <- blurImage(readImage(paste0(dir,"/",w), resize = 0.5),
                     blur_function = "medianblur", n = 3, plotting = FALSE)
    # basic recolorize2 step:
    rc1 <- recolorize2(rc0, bins = c(3,4,3), cutoff = 25, plotting = FALSE)
    
    # drop minor colors
    # rc2 <- thresholdRecolor(rc1, plotting = FALSE)
    # and recluster again
    if(length(rc1$sizes) > 1){
      rc1 <- recluster(rc1, cutoff = 25, plot_final = FALSE, plot_hclust = FALSE)
    }
    
    recolorize_to_png(rc1,paste0(write_dir,"/",w))
    print(counter + start_index)
    counter <- counter + 1
  }
}
row <- wing_colors[1,]
plotRGBColor(c(row$col_1_r,row$col_1_g,row$col_1_b),max=1)
plotRGBColor(c(row$col_2_r,row$col_2_g,row$col_2_b),max=1)
plotRGBColor(c(row$col_3_r,row$col_3_g,row$col_3_b),max=1)
plotRGBColor(c(row$col_4_r,row$col_4_g,row$col_4_b),max=1)
plotRGBColor(c(row$col_5_r,row$col_5_g,row$col_5_b),max=1)
plotRGBColor(c(row$col_6_r,row$col_6_g,row$col_6_b),max=1)

#plotRGBColor(c(row$col_6_r,row$col_6_g,row$col_6_b),max=1)

#2651
recolorizeWingSegmentDir(dir,write_dir,4530)
