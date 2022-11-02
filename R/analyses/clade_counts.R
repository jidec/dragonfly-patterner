# counting and plotting species and genus diversity within genera and families 
# getting a list of genera indices within genus list

data <- inat_records
library(dplyr)
test <- data %>%
    group_by(genus) %>%
    summarise(n_species = length(unique(species)))

sum(test$n_species)
hist(test$n_species)

test <- data %>%
    group_by(family) %>%
    summarise(n_species = length(unique(species)), n = n())


table(data$family)
sum(test$n_species)
hist(test$n_species)

aesh_sp <- data[data$family=="Aeshnidae",]$species
table(aesh_sp)

aesh_genera <- unique(data[data$family=="Aeshnidae",]$genus)

gomph_genera <- unique(inat_records[inat_records$family=="Gomphidae",]$genus)

match(genera,genus_list$X0) - 1
aesh_genera 
genus_list - 1
genus_list <- read.csv("E:/dragonfly-patterner/data/other/genus_list.csv")
