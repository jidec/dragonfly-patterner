
#ref_path <- "E:/dragonfly-patterner/data/patterns/gomph_grouped_5000/INAT-82301046-2_pattern.png"
#pattern_dir <- "E:/dragonfly-patterner/data/patterns/gomph_grouped_5000"
#clust_val_channel1 <- 0.7254902

discretizedPCA <- function(pattern_dir,ref_path,clust_val_channel1){
  library(png)
  library(RNiftyReg)
  library(r)
  print("Loading patterns...")
  # load all patterns
  imgs <- list()
  for(f in list.files(pattern_dir,full.names = TRUE)){
    imgs <- append(imgs,list(readPNG(f)))
  }
  # load reference 
  ref <- readPNG(ref_path)
  
  # align all patterns to reference
  #i = 1
  print("Aligning patterns to reference using rNiftyReg...")
  for(i in 1:length(imgs)){
    imgs[[i]] <- niftyreg(imgs[[i]], ref)$image
    if(i%%10 == 0){
      print(i)
    }
  }
  
  cluster_masks <- list()
  
  print("Getting raster masks for cluster")
  # get raster masks for cluster
  for(i in 1:length(imgs)){
    
    # convert to arr
    arr <- as.array(imgs[[i]])
    # save original dims
    og_dim <- dim(arr)
    
    # reshape to pixels
    dim(arr) <- c(dim(arr)[1] * dim(arr)[2],4)
    # get 1/NA mask of where cluster occurs
    clust_mask <- abs(arr[,1] - clust_val_channel1) < 0.0001
    clust_mask[clust_mask == TRUE] <- 1
    clust_mask[clust_mask == FALSE] <- NA
    
    # reshape back to 2d array
    dim(clust_mask) <- c(og_dim[1],og_dim[2])
    
    # convert to RasterLayer and add to list
    library(raster)
    clust_mask <- raster(clust_mask)
    cluster_masks <- append(cluster_masks,list(clust_mask))
  }
  
  pop1 <- c('BC0077','BC0071')
  pop2 <- c('BC0050','BC0049','BC0004')
  popList <- list('None')
  colList <- c("red")
  
  pcaOut <- patPCA(cluster_masks, popList, colList, plot = TRUE,plotCartoon = TRUE,plotChanges = TRUE,refImage = raster::stack(ref_path))
  
  pcaOut <- patPCA(cluster_masks, popList, colList, plot = TRUE, plotType = 'points', plotChanges = TRUE, 
                   PCx = 1, PCy = 2, plotCartoon = TRUE, refShape = 'target', outline = outline_IMG_2600, flipOutline = 'y', 
                   imageList = imageListGala, cartoonID = 'IMG_2600', normalized = TRUE, cartoonFill = 'black', 
                   cartoonOrder = 'under', legendTitle = 'Predicted')
}