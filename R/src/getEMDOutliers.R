getEMDOutliers <- function(image_dir, outlier_quantile=0.97, return_raw_emds=FALSE){
  library(imager)
  print("Loading images...")
  imgs <- load.dir(image_dir)
  lightness_mats <- list()
  for(i in 1:length(imgs)){
    r <- as.matrix(as.array(imgs[[i]])[,,,1])
    g <- as.matrix(as.array(imgs[[i]])[,,,2])
    b <- as.matrix(as.array(imgs[[i]])[,,,3])
    l <- (r + g + b) / 3
    lightness_mats <- append(lightness_mats,list(l))
    print(i)
  }
  
  names(lightness_mats) <- names(imgs)
  print("Calculating EMDs (will take awhile)...")
  emds <- matrix(nrow=length(lightness_mats),ncol=length(lightness_mats))
  for(m in 1:length(lightness_mats)){
    for(m2 in 1:length(lightness_mats)){
      emds[m,m2] <- emd(lightness_mats[[m]],lightness_mats[[m2]])
    }
    print(m)
  }
  
  sum_emds <- rowMeans(emds)
  is_emd_outlier_vect <- sum_emds > quantile(sum_emds,outlier_quantile)
  names(is_emd_outlier_vect) <- list.files(image_dir)
  
  if(return_raw_emds){
    return(emds)
  }
  else{
    return(is_emd_outlier_vect) 
  }
}

getEMDOutliers("E:/dragonfly-patterner/data/patterns/gomph_grouped_5000")
