
extractMeanColor <- function(img_dir,start_index=1){
  library(imager)
  df <- data.frame()
  paths <- list.files(img_dir, full.names = TRUE)
  paths <- paths[start_index:length(paths)]
  i <- 0
  for(path in paths){
    if(i %% 1000 == 0){
      print(i)
    }
    i <- i + 1
    #print(i)
    #print(path)
    if(file.exists(path) & endsWith(path,".png")){
      img <- load.image(path)
      arr <- as.array(img)
      #print(dim(arr))
      #print(dim(arr)[0])
      if(dim(arr)[1] > 1){
        # reshape to pixels
        dim(arr) <- c(dim(arr)[1] * dim(arr)[2],4)
        
        # remove black transparent background pixels 
        arr <- arr[(arr != c(0,0,0,0))[,1],]
        
        # create row for image
        row <- c(path)
        
        # add mean color to row
        mean_rgb <- c(mean(arr[,1]), mean(arr[,2]), mean(arr[,3]))
        row <- c(row,mean_rgb)
        
        df <- rbind(df,row) 
    }
    }
  }
  return(df)
}
