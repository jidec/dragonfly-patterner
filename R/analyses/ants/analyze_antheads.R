gasters <- extractMeanColor("D:/ant-patterner/data/segments/gaster")
colors <- colors2
source("src/extractMeanColor.R")
colors2 <- extractMeanColor("D:/ant-patterner/data/segments/grey_adj")
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
