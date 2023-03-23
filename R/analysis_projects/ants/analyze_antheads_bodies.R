# we expect:
# ants with stronger defenses will have more distinct color patterns
# code clades based on the intensity of their anti-predator defenses
# spines and sting 
# https://onlinelibrary.wiley.com/doi/full/10.1111/evo.13117

# result - not the case! ant genera do not have aposematic color
# import color data
#source("src/extractMeanColor.R")
#thorax <- extractMeanColor("D:/ant-patterner/data/segments/thorax")
thorax <- read.csv("D:/ant-patterner/thorax_colors.csv")
colnames(thorax) <- c("1","recordID", "r","g","b")
gaster <- read.csv("D:/ant-patterner/gaster_colors_adj.csv")
head <- read.csv("D:/ant-patterner/AntWeb_specimen_head_colors.csv")

# setup color data 
gaster <- data.frame(cbind(gaster$recordID,gaster$r,gaster$g,gaster$b))
thorax <- data.frame(cbind(thorax$recordID,thorax$r,thorax$g,thorax$b))
colnames(gaster) <- c("recordID","gaster_r","gaster_g","gaster_b")
colnames(thorax) <- c("recordID","thorax_r","thorax_g","thorax_b")
library(stringr)
thorax$recordID <- str_split_fixed(str_split_fixed(thorax$recordID,"/",n=6)[,6],"_",n=5)[,1]

# merge all body part colors
ants <- merge(head,gaster,by="recordID")
ants <- merge(ants,thorax,by="recordID")

# fix numeric
library(dplyr)
ants <- ants %>% mutate_at(c('r', 'g','b','thorax_r','thorax_g','thorax_b','gaster_r','gaster_g','gaster_b'), as.numeric)

# create var for sum of color difference between parts 
ants$htg_diff <- abs(ants$r - ants$thorax_r) + abs(ants$r - ants$gaster_r) + abs(ants$thorax_r - ants$gaster_r) +
  abs(ants$g - ants$thorax_g) + abs(ants$g - ants$gaster_g) + abs(ants$thorax_g - ants$gaster_g) +
  abs(ants$b - ants$thorax_b) + abs(ants$b - ants$gaster_b) + abs(ants$thorax_b - ants$gaster_b)

# read traits 
library(readxl)
traits <- read_excel("D:/ant-patterner/AntGeneraDefensiveTraits.xlsx")
colnames(traits) <- traits[2,]
traits <- traits[-(1:2),]
traits$has_spines <- traits$`Spines (0=absent; 1=spine of rank 2 present anywhere on mesosoma or petiole); All spinescence data based on Antweb images (accessed December 2014) unless otherwise noted in References` == 1
traits$has_sting <- traits$`Sting (0=absent, 1=present)` == 1
traits <- data.frame(cbind(traits$has_sting,traits$has_spines,traits$Genus))
colnames(traits) <- c("has_sting","has_spines","genus")

ants$species <- paste(ants$genus,ants$species)
workers <- filter(ants,caste=="worker")
workers <- workers %>%
  group_by(species) %>%
  summarise(htg_diff = mean(htg_diff), n = n())

workers[workers$htg_diff >= quantile(workers$htg_diff,0.98),]

# merge traits into color data
#ants <- merge(ants,traits,by="genus")
workers <- merge(workers,traits,by="genus")
workers$htg_diff <- scale(workers$htg_diff)
quantile(workers$htg_diff,0.5)
workers[workers$htg_diff >= quantile(workers$htg_diff,0.5),]
model <- lm(htg_diff ~ has_sting + has_spines, data = workers)
summary(model)

anova <- aov(htg_diff ~ has_sting + has_spines, data = workers)

summary(anova)
tukey <-TukeyHSD(anova)

tukey

workers <- filter(ants,caste=="worker")
model <- lm(htg_diff ~ has_sting + has_spines, data = workers)
summary(model)
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