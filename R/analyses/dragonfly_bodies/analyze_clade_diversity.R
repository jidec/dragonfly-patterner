# counting and plotting species and genus diversity within genera and families 

library(dplyr)
genera <- gbif_records %>%
    group_by(genus) %>%
    summarise(n_species_in_genus = length(unique(species)))

families <- gbif_records %>%
    group_by(family) %>%
    summarise(n_species_in_family = length(unique(species)), n = n())

View(genera)
View(families)
