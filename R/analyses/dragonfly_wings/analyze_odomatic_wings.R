library(ape)
library(stringr)

# extract color stats from discretized wings
source("src/extractSimpleColorStats.R")
wing_colors <- extractSimpleColorStats("D:/wing-color/data/patterns")

# load records and merge 
wings <- read.csv("D:/wing-color/data/records.csv")
wings$county <- str_to_title(paste(wings$County,wings$State))
wings$recordID <- wings$uniq_id
wings$species <- wings$Species.name
wings <- merge(wing_colors,wings,all=FALSE)

# load odonate tree
odonate_tree <- ape::read.tree("E:/dragonfly-patterner/R/data/odonata.tre")
library(stringr)
odonate_tree$tip.label <-  str_replace(odonate_tree$tip.label,"_"," ")
data_phy <- removeDataPhyloMissing(wings,odonate_tree)
wings <- data_phy[[1]]
wings_phy <- data_phy[[2]]

wings <- wings %>%
    group_by(Species.name) %>%
    summarise(trait = mean(col_1_prop))

colnames(wings) <- c("clade","trait")
plotPhyloEffects(wings,phy)

# sdms
sdms <- read.csv("D:/wing-color/data/Nearctic_ModelResults_5KM.csv",nrows=100000)
View(sdms)

# counties
library(housingData)
counties <- housingData::geoCounty
counties$county <- str_to_title(paste(counties$rMapCounty,counties$rMapState))
wings <- merge(wings,counties)

l <- lm(col_1_prop ~ Sex, data=wings) #as.numeric(merged$col_1_prop),merged$lat)
summary(l)
hist(as.numeric(wings$col_1_prop),breaks=190)

# merge wings and bodies by species
