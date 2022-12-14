loadGenusDLRecords <- function(write=FALSE){
  # importing genus records 
  gen_recs <- list.files(path = "E:/dragonfly-patterner/data/other/genus_download_records",  # Identify all CSV files
                         pattern = "*.csv", full.names = TRUE)
  gen_recs <- gen_recs[!grepl("log",gen_recs)]
  
  library(dplyr)
  library(readr)
  gen_recs <- gen_recs %>% lapply(read_csv)
  
  for (g in 1:length(gen_recs)){
    gen_recs[[g]]$obs_id <- as.character(gen_recs[[g]]$obs_id)
    gen_recs[[g]]$usr_id <- as.character(gen_recs[[g]]$usr_id)
    gen_recs[[g]]$date <- as.character(gen_recs[[g]]$date)
    gen_recs[[g]]$latitude <- as.double(gen_recs[[g]]$latitude)
    gen_recs[[g]]$longitude <- as.double(gen_recs[[g]]$longitude)
    gen_recs[[g]]$img_cnt <- as.double(gen_recs[[g]]$img_cnt)
    gen_recs[[g]]$img_id <- as.character(gen_recs[[g]]$img_id)
    gen_recs[[g]]$width <- as.double(gen_recs[[g]]$width)
    gen_recs[[g]]$height <- as.double(gen_recs[[g]]$height)
  }
  
  gen_recs <- gen_recs %>% bind_rows
  
  gen_recs$catalogNumber <- gen_recs$obs_id
  source("src/mergeImagesToRecords.R")
  bound_records <- mergeImagesToDLRecords("E:/dragonfly-patterner/data/all_images",gen_recs,"has_downloaded_image")
  
  bound_records$genus <- str_split_fixed(bound_records$taxon, " ",2)[,1]
  # USA
  bound_records <- filter(bound_records,latitude>24.39,latitude<49.38,longitude>(-124),longitude<(-66.88))
  bound_records$recordID <- paste0("INAT-",bound_records$recordID)
  library(stringr)
  bound_records$imageID <- paste0("INAT-",str_replace(bound_records$file_name,".jpg",""))
  bound_records$species <- bound_records$taxon
  if(write){
    write.csv(bound_records,"E:/dragonfly-patterner/data/records.csv")
  }
  return(bound_records)
}