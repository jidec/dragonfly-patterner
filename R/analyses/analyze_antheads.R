
source("src/extractMeanColor.R")
colors <- extractMeanColor("E:/ant-patterner/data/segments")
colnames(colors) <- c("recordID","r","g","b")

library(stringr)
col <- str_split(colors$recordID,"/")

l <- str_extract(mean_colors$X.E..ant.patterner.data.segments.08costa.0161_h_segment.png., "[^/]*$") # weird colname
l <- str_extract(l,"^.*?(?=_)")
colors$recordID <- l

ant_records <- read.csv("E:/ant-patterner/data/antweb_records.csv")
data <- merge(colors,ant_records,by="recordID")

library(dplyr)
data <- data[!duplicated(data$recordID),]
data$r <- as.numeric(data$r)
data$b <- as.numeric(data$b)
data$g <- as.numeric(data$g)
data <- data[!is.na(data$g),]
nrow(data)
d <- rgb2hsv(data$r,data$g,data$b,maxColorValue = 1)
data <- cbind(data,d)

test <- inner_join(colors,ant_records)
test <- test[!is.na(test$r),]

write.csv(test2,"specimen_colors_unadj.csv")