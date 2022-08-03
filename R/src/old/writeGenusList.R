
# create a list of genera from all Odonata csv

writeGenusList <- function(proj_root="../.."){
  
  library(dplyr)
  library(stringr)
  
  data <- read.csv(paste0(proj_root, "/data/inat_data.csv"),header=TRUE,row.names=NULL,sep=",")
  genera <- unique(data$scientificName)
  genera <- str_split_fixed(data$species, " ", n=2)
  genera <- unique(genera[,1])
  write.csv(genera, paste0(proj_root, "/data/other/genus_list.csv"))
  print("Wrote new genus list .csv to data/misc folder")
}