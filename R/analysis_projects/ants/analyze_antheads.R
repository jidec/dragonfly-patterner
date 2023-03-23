
#source("src/extractMeanColor.R")
#gasters <- extractMeanColor("D:/ant-patterner/data/segments/gaster")

# we expect:
# ants with stronger defenses will have more distinct color patterns
# code clades based on the intensity of their anti-predator defenses
# spines and sting 
# https://onlinelibrary.wiley.com/doi/full/10.1111/evo.13117

gasters <- read.csv("D:/ant-patterner/gaster_colors_adj.csv")
heads <- read.csv("D:/ant-patterner/AntWeb_specimen_head_colors.csv")

gasters$part <- "gaster"
heads$part <- "head"

gasters <- data.frame(cbind(gasters$part,gasters$recordID,gasters$r,gasters$g,gasters$b))
colnames(gasters) <- c("part","recordID","gaster_r","gaster_g","gaster_b")

ants <- merge(heads,gasters,by="recordID")

ants$gaster_r <- as.numeric(ants$gaster_r)
ants$gaster_g <- as.numeric(ants$gaster_g)
ants$gaster_b <- as.numeric(ants$gaster_b)
ants$gaster_lightness <- (ants$gaster_r + ants$gaster_g + ants$gaster_b) /3
ants$lightness <- (ants$r + ants$g + ants$b) /3

cor.test(ants$lightness,ants$gaster_lightness)

ants$hg_lightness_diff <- abs(ants$lightness - ants$gaster_lightness)
hist(ants$hg_lightness_diff,breaks=100)
high_diff <- filter(ants,hg_lightness_diff >= 0.3)
visualizeID(high_diff$recordID[203],repo_path="D:/ant-patterner",data_folder="all_images",id_suffix="_d",img_ext=".jpg")
# neat cor
cor.test(ants$hg_lightness_diff,as.numeric(ants$gaster_lightness))
# keep in mind wings over gaster make them light

hist(table(ants$part.x))
heads$part <- "head"
ants <- rbind(heads,gasters)
ants$lightness <- (ants$r + ants$g + ants$b) /3

m <- lm(lightness ~ part, ants)
ants$lightness <- scale(ants$lightness)

library(dplyr)
hist(filter(ants,part=="gaster")$lightness)
hist(filter(ants,part=="head")$lightness)

plotRGBColor(c(mean(gasters$r),mean(gasters$g),mean(gasters$b)),max=1)
plotRGBColor(c(mean(heads$r),mean(heads$g),mean(heads$b)),max=1)

cor.test()

mean(gasters)
summary(m)
colors <- gasters
colnames(colors) <- c("recordID","r","g","b")
saved <- colors

library(stringr)
col <- str_split(colors$recordID,"/")

l <- str_extract(colors$recordID, "[^/]*$") # weird colname
l <- str_extract(l,"^.*?(?=_)")
colors$recordID <- l

ant_records <- read.csv("E:/ant-patterner/data/records.csv")
data <- merge(colors,ant_records,by="recordID")

library(dplyr)
data <- data[!duplicated(data$recordID),]
nrow(data)
data$r <- as.numeric(data$r)
data$b <- as.numeric(data$b)
data$g <- as.numeric(data$g)
data <- data[!is.na(data$g),]
nrow(data)
d <- rgb2hsv(data$r,data$g,data$b,maxColorValue = 1)
d <- t(d)

data <- cbind(data,d)

write.csv(data,"../gaster_colors_adj.csv")
