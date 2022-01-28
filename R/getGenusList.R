
# create a list of genera from all Odonata csv

getGenusList <- function(){
  
  library(dplyr)
  library(stringr)
  
  data <- read.csv("../data/inat_odonata_usa.csv",header=TRUE,row.names=NULL,sep="\t")
  genera <- unique(data$scientificName)
  genera <- str_split_fixed(data$species, " ", n=2)
  genera <- unique(genera[,1])
  genera
  write.csv(genera, "../data/genus_list.csv")
}