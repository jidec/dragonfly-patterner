# path = "../experiments/odo_seg_analyzer/images/4155610_198_discrete.png"

extractSimpleColorStats <- function(img_paths){
  library(imager)
  
  df <- data.frame()
  for(path in img_paths){
    print(path)
    img <- load.image(path)
    arr <- as.array(img)
  
    # reshape to pixels
    dim(arr) <- c(dim(arr)[1] * dim(arr)[2],4)
    
    # remove black background pixels 
    arr <- arr[(arr != c(0,0,0,0))[,1],]
    
    # create row for image
    row <- c()
    # add mean color to row
    mean_rgb <- c(mean(arr[,1]), mean(arr[,2]), mean(arr[,3]))
    row <- c(row,mean_rgb)
    # get unique colors
    colors <- unique(arr)
    # for every color
    for(c in 1:nrow(colors)){
      color <- colors[c,]
      
      # calculate proportion of pixels
      prop <- sum(arr[,1] ==  color[1] & arr[,2] == color[2] & arr[,3] == color[3]) / nrow(arr)
      
      # add the color and prop color to row 
      row <- c(row,color,prop)
    }
    df <- rbind(df,row)
  }
}

data <- read.csv("analyses/specimen_colors_unadj.csv")
unique(data$recordID)


