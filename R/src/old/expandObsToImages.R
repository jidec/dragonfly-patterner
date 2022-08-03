
# expand the iNat data such that each row is an image instead of an observation
expandObsToImages() <- function()
{
  data <- read.csv("../data/image_obs_metadata.csv",header=TRUE,row.names=1,sep=",")
  image_names <- list.files("../data/all_images")
  image_names <- sub(".jpg","",image_names)
  library(stringr)
  image_ids <- word(image_names,1,sep = "_")
  
  out <- data[FALSE,]
  j <- 1
  #i <- 1
  # for every row in data
  for(i in 1:nrow(data))
  {
    id <- data$catalogNumber[i]
    matches <- image_names[image_ids == id]
    if(length(matches) > 0){
      for(m in 1:length(matches))
      {
        # copy row from data 
        out[j,1:19] <- data[i,]
        # add match name
        out$imageName[j] <- matches[m]
        j <- j + 1
      }
    }
    print(i)
  }
  
  return(out)
}