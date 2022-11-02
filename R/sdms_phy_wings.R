library(ape)
library(stringr)

records <- read.csv("D:/wing-color/data/records.csv")
records$county <- str_to_title(paste(records$County,records$State))
records$recordID <- records$uniq_id
records$species <- records$Species.name
records <- merge(wing_colors,records,all=FALSE)

#t <- wing_colors
#cols <- 2:ncol(t)
#t[,cols] <- sapply(t[,cols],as.numeric

#tree
odonate_tree <- ape::read.tree("E:/dragonfly-patterner/R/data/odonata.tre")
library(stringr)
odonate_tree$tip.label <-  str_replace(odonate_tree$tip.label,"_"," ")

data_phy <- removeDataPhyloMissing(records,odonate_tree)
records <- data_phy[[1]]
records$col_1_prop <- as.numeric(records$col_1_prop)
phy <- data_phy[[2]]

records <- records %>%
    group_by(Species.name) %>%
    summarise(trait = mean(col_1_prop))

colnames(records) <- c("clade","trait")
plotPhyloEffects(records,phy)

# sdms
sdms <- read.csv("D:/wing-color/data/Nearctic_ModelResults_5KM.csv",nrows=100000)
View(sdms)

# counties
library(housingData)
counties <- housingData::geoCounty
counties$county <- str_to_title(paste(counties$rMapCounty,counties$rMapState))

data <- merge(records,counties)

#merged <- merged[!duplicated(merged$recordID),]

#merged$col_1_prop <- as.numeric(merged$col_1_prop)
#merged$col_5_prop <- as.numeric(merged$col_5_prop)
#length(unique(merged$Species.name))
#sum(merged$Sex == "M")

data$lightness <- rowMeans(data[,3:5])

l <- lm(col_1_prop ~ Sex, data=data) #as.numeric(merged$col_1_prop),merged$lat)
summary(l)
hist(as.numeric(data$col_1_prop),breaks=190)
