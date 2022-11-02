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

bound_records <- mergeImagesToDLRecords("E:/dragonfly-patterner/data/all_images",gen_recs,"has_downloaded_image")

bound_records$genus <- str_split_fixed(bound_records$taxon, " ",2)[,1]
bound_records <- filter(bound_records,latitude>24.39,latitude<49.38,longitude>(-124),longitude<(-66.88))

table(bound_records$genus)
aesh_records <- bound_records[!is.na(match(bound_records$genus,aesh_genera)),]
gomph_records <- bound_records[!is.na(match(bound_records$genus,gomph_genera)),]

bound_records$recordID <- paste0("INAT-",bound_records$recordID)

is_aesh <- !is.na(match(bound_records$genus,aesh_genera))
bound_records$is_aesh <- is_aesh

#is_aesh[is_aesh == FALSE] <- ""
#is_aesh[is_aesh == TRUE] <- "Aeshnidae"
#bound_records$family <- is_aesh

is_gomph <- !is.na(match(bound_records$genus,gomph_genera))
bound_records$is_gomph <- is_gomph

# SAVE
# maybe rename old records first 
bound_records$recordID <- bound_records$catalogNumber
write.csv(bound_records,"E:/dragonfly-patterner/data/records.csv")

sum(bound_records$genus == "Anax")

table(bound_records$genus)
table(bound_records$taxon)
anax <- filter(bound_records,genus == "Anax")

table(anax$taxon)

bound_records$imageID <- bound_records$catalogNumber
test <- bound_records %>%
  group_by(genus) %>%
  summarise(n_species = length(unique(taxon)), n = n())
