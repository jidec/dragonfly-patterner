# 

table(records$genus)
aesh_records <- bound_records[!is.na(match(bound_records$genus,aesh_genera)),]
gomph_records <- bound_records[!is.na(match(bound_records$genus,gomph_genera)),]

records <- records %>%
  group_by(genus) %>%
  summarise(n_species_in_genus = length(unique(taxon)), n = n())