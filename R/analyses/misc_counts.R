# count number of a specific genus
data <- read.csv("../data/inat_odonata_usa.csv",header=TRUE,row.names=NULL,sep="\t")
library(dplyr)


genus_obs <- dplyr::filter(data, genus == "Stylurus")
unique(genus_obs$species) # species in genus 
