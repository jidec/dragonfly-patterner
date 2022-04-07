
# create a list of genera from all Odonata csv

writeGenusList <- function(){
  
  library(dplyr)
  library(stringr)
  
  data <- read.csv("../data/inat_odonata_usa.csv",header=TRUE,row.names=NULL,sep=",")
  genera <- unique(data$scientificName)
  genera <- str_split_fixed(data$species, " ", n=2)
  genera <- unique(genera[,1])
  write.csv(genera, "../pipeline/0_image_downloading/pylib/genus_list.csv")
}