
#img_dir = "E:/dragonfly-patterner/data/segments"
#lightness_cutoff=0.9
#percent_past_cutoff=0.1

simpleInferBad <- function(img_dir, lightness_cutoff=0.9, percent_past_cutoff=0.1){
  print("Loading images...")
  library(imager)
  images <- load.dir(img_dir)
  
  # resize all to 500 so that EMD works
  for(i in 1:length(images)){
    images[[i]] <- resize(images[[i]],size_x=500)
  }
  
  l_channels <- list()
    
  print("Checking if glared")
  is_glared_vect <- c()
  for(img in images){
    #l <- (img[,,,1] + img[,,,2] + img[,,,3]) / 3
    l <- (as.matrix(as.array(img)[,,,1]) + as.matrix(as.array(img)[,,,2]) + as.matrix(as.array(img)[,,,3])) / 3
    l_channels <- append(l_channels,list(l))
    is_glared <- (sum(l > lightness_cutoff) / length(l)) > percent_past_cutoff
    is_glared_vect <- c(is_glared_vect,is_glared)
  }
  
  library(emdist)
  emds <- matrix(nrow=length(l_channels),ncol=length(l_channels))
  for(m in 1:length(l_channels)){
    for(m2 in 1:length(l_channels)){
      emds[m,m2] <- emd(l_channels[[m]],l_channels[[m2]])
    }
  }
  sum_emds <- rowMeans(emds)
  is_emd_outlier <- sum_emds > quantile(sum_emds,0.97)
  
  names(is_emd_outlier) <- list.files(img_dir)
  names(is_glared_vect) <- list.files(img_dir)
  
  return(cbind(is_emd_outlier,is_glared_vect))
}

simple_bad <- simpleInferBad("E:/dragonfly-patterner/data/segments",percent_past_cutoff = 0.05)
simple_bad[simple_bad == TRUE]


getGlaredByLightnessCutoff <- function(image_dir, lightness_cutoff=0.9, percent_past_cutoff=0.1){
    
  print("Loading images...")
  library(imager)
  images <- load.dir(image_dir)
  
  l_channels <- list()
  
  print("Checking if glared")
  is_glared_vect <- c()
  for(img in images){
    l <- (img[,,,1] + img[,,,2] + img[,,,3]) / 3
    is_glared <- (sum(l > lightness_cutoff) / length(l)) > percent_past_cutoff
    is_glared_vect <- c(is_glared_vect,is_glared)
  }
  
  names(is_glared_vect) <- list.files(image_dir)
  
  return(is_glared_vect)
}
