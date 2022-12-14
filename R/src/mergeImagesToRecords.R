
#dir = "E:/dragonfly-patterner/data/all_images"
#records=inat_records
#new_records_colname = "has_downloaded_image"

# use to get records for data at different stages in the pipeline
mergeImagesToRecords <- function(dir, records, new_records_colname){
    library(stringr)
    library(dplyr)
    print(paste("Number of records", nrow(records)))
    imgs <- list.files(path = dir)
    print(paste("Number of imgs (multiple per record)", length(imgs)))
    inat_ids <- unique(str_split_fixed(imgs, "-", n =3)[,2])
    print(paste("Number of unique ids (record numbers) in images", length(inat_ids)))
    df <- cbind(rep(TRUE,length(inat_ids)),inat_ids)
    colnames(df) <- c(new_records_colname,"catalogNumber")
    merged <- merge(records,df,by="catalogNumber",all= TRUE)
    merged[new_records_colname][is.na(merged[new_records_colname])] <- FALSE
    merged["gbifID"][is.na(merged["gbifID"])] <- -1
    print(paste("Number of downloaded:",nrow(filter(merged, has_downloaded_image == TRUE))))
    print(paste("Number of downloaded with matching record:",nrow(filter(merged, has_downloaded_image == TRUE, gbifID!=-1))))
    
    return(merged)
}

# use to get records for data at different stages in the pipeline
mergeImagesToDLRecords <- function(dir, records, new_records_colname){
  library(stringr)
  library(dplyr)
  print(paste("Number of records", nrow(records)))
  imgs <- list.files(path = dir)
  print(paste("Number of imgs (multiple per record)", length(imgs)))
  inat_ids <- unique(str_split_fixed(imgs, "-", n =3)[,2])
  print(paste("Number of unique ids (record numbers) in images", length(inat_ids)))
  df <- cbind(rep(TRUE,length(inat_ids)),inat_ids)
  colnames(df) <- c(new_records_colname,"catalogNumber")
  merged <- merge(records,df,by="catalogNumber",all= TRUE)
  merged[new_records_colname][is.na(merged[new_records_colname])] <- FALSE
  merged["usr_id"][is.na(merged["usr_id"])] <- -1
  print(paste("Number of downloaded:",nrow(filter(merged, has_downloaded_image == TRUE))))
  print(paste("Number of downloaded with matching record:",nrow(filter(merged, has_downloaded_image == TRUE, usr_id!=-1))))
  
  return(merged)
}