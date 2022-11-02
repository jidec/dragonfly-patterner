
# verify that all records are from US
table(inat_records$countryCode)

# verify that all ids are unique
length(unique(inat_records$catalogNumber))

# count number of Bas records 
sum(inat_records$genus == "Basiaeschna")
# 690 Bas USA records

# 440 are in Canada 
test <- filter(merged,has_downloaded_image == TRUE,genus=="Basiaeschna")

# count number of bas image records
bas_obs_ids <- unique(bas_dl_records$obs_id)
length(unique(bas_dl_records$obs_id))
# 497 were downloaded

# make sure that all in bas_dl_records were actually downloaded
imgs <- list.files(path = dir)
inat_ids <- unique(str_split_fixed(imgs, "-", n =3)[,2])
match(bas_obs_ids,inat_ids)

length(unique(bas_dl_records$obs_id)) 

# merge images to records
#inat_records$catalogNumber <- as.character(inat_records$catalogNumber)
merged <- mergeImagesToRecords(dir="E:/dragonfly-patterner/data/all_images", records=inat_records,
                               new_records_colname = "has_downloaded_image")
merged$has_downloaded_image[is.na(merged$has_downloaded_image)] <- FALSE

genus <- "Anax"
print(paste("Number of records for genus:",nrow(filter(inat_records, genus== "Anax"))))
print(paste("Number of downloaded for genus:",nrow(filter(merged, genus== "Anax", has_downloaded_image == TRUE))))

bas$has_downloaded_image
nrow(filter(inat_records, genus== "Anax"))
# get genus with downloaded images
# all from records or imgs with genus
bas <- filter(merged, genus == "Basiaeschna")
# from imgs 
bas2 <- bas[bas$has_downloaded_image == TRUE,]

test <- cbind(bas$catalogNumber,bas$has_downloaded_image)
library(dplyr)
ax <- filter(merged, genus == "Anax")
ax2 <- ax[ax$has_downloaded_image == TRUE,]

library(stringr)
inat_ids <- str_split_fixed(imgs, "-", n =3)[,2]
dir <- "E:/dragonfly-patterner/data/all_images"
imgs <- getImageRecords("E:/dragonfly-patterner/data/all_images")
new_records_colname <- "has_downloaded_image"
merged <- merge(df,data,by="catalogNumber",all.y=TRUE)

sum(merged$has_downloaded_image == TRUE)
merged$has_downloaded_image[merged$has_downloaded_image != TRUE] <- FALSE
sum(c(TRUE,TRUE,FALSE))

sum(is.na(merged$gbifID))
merged$catalogNumber
paste(merged$occurrenceID,merged$catalogNumber)
merged$occurrenceID
merged$has_downloaded_image[is.na(merged$has_downloaded_image)] <- FALSE 

nrow(filter(inat_records,genus=="Basiaeschna"))

bas_dl_records <- read.csv("E:/dragonfly-patterner/data/other/genus_download_records/iNat_images-Basiaeschna-records_split/1.csv")
length(unique(bas_dl_records$obs_id))

# a few downloaded images should NOT have matches in records as they are new or in a non USA part of the bounding box
# but all records should have a match, given that records are USA dflies and all of those should have been downloaded
#       EXCEPT observations in hawaii, alaska, etc which are not within the bounding box and thus not downloaded 