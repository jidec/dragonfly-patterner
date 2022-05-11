
# rename images downloaded from Odonata Central 
# replaces dashes with underscores
prepOCImageData <- function(relative_root="../..")
{
  library(stringr)
  oc_data_keeps <- read.csv("../data/odonata_central.csv")
  # remove files already in iNaturalist
  oc_data_keeps <- oc_data_keeps[str_detect(oc_data_keeps$Specimen.Notes,"iNaturalist",negate=TRUE),]
  oc_data_keeps <- oc_data_keeps[str_detect(oc_data_keeps$Specimen.Notes,"inaturalist",negate=TRUE),]
  # keep files with high confidence
  oc_data_keeps <- oc_data_keeps[oc_data_keeps$ID.Confidence=="High",]
  # keep files with photos
  oc_data_keeps <- oc_data_keeps[oc_data_keeps$Record.Type=="Photo",]
  
  #rename files replacing dash with underscore
  oc_files <- list.files("../data/all_images",pattern="-")
  oc_ids <- sub("-.*", "", oc_files)
  # remove ids are any ids not in keeps
  remove_ids <- setdiff(oc_ids,oc_data_keeps$OC..)
  # rename oc files
  oc_renamed <- str_replace(oc_files,"-","_")
  file.rename(oc_files,oc_renamed)
  # remove non keeps
  remove_ids <- paste0(remove_ids,"_")
  remove_files <- list.files("../data/all_images",pattern=remove_ids)

  # delete remove files
  unlink(paste0("../data/all_images/",remove_files))
  
}