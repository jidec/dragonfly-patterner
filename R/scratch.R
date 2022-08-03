
img_dir = "E:/ant-patterner/data/segments"
colors <- extractMeanColor("E:/ant-patterner/data/segments")
colnames(colors) <- c("recordID","r","g","b")
library(stringr)
col <- str_split(colors$recordID,"/")

l <- str_extract(mean_colors$X.E..ant.patterner.data.segments.08costa.0161_h_segment.png., "[^/]*$") # weird colname
l <- str_extract(l,"^.*?(?=_)")
colors$recordID <- l

data <- merge(colors,ant_records,by="recordID")

library(dplyr)
data <- data[!duplicated(data$recordID),]
data$r <- as.numeric(data$r)
data$b <- as.numeric(data$b)
data$g <- as.numeric(data$g)
data <- data[!is.na(data$g),]
data2$h
is.na()
length(data$g)
length(complete.cases(data$g))
d <- rgb2hsv(data$r,data$g,data$b,maxColorValue = 1)
data <- cbind(data,d)

t <- data.frame(t(t))
test <- test[t ]
test$
  sum(is.na(test$r))
  
  test <- inner_join(colors,ant_records)
View(test)
View(c)
test <- test[!is.na(test$r),]
ant_records <- read.csv("E:/ant-patterner/data/antweb_records.csv")

write.csv(test2,"specimen_colors_unadj.csv")

test <- read.csv("../data/antweb_raw.csv", sep = "\t",header=F)
col <- test$V39
sum(col == "usnment01125594")

library(stringr)

sum(grepl("usnment",col))

col[grepl("usnment",col)]
library(stringr)
files <- list.files(path="../data/new_images/antweb-images/antweb-images")
split <- str_split_fixed(files,"_",4)
split <- split[split[,2] != "",]
split <- paste(split[,1],split[,2])
split
split <- split[grepl(" h", split)]
length(unique(split))
length(unique(split))
combined <- c(match(image_names,data_names),match(image_names_adj,data_names))
length(unique(combined))
unique(combined)

col

antweb_specimen_codes <- image_names

write.csv(antweb_specimen_codes,"antweb_specimen_codes3.csv")

write.csv(matrix(antweb_specimen_codes, nrow=1), file ="antweb_specimen_codes2.csv", row.names=FALSE,col.names = FALSE)

library(stringr)
antweb_specimen_codes[antweb_specimen_codes == "238376:mem 207971"]
antweb_specimen_codes[grepl("mem",antweb_specimen_codes)]

antweb_specimen_codes <- str_replace(antweb_specimen_codes," ","")

antweb_specimen_codes <- antweb_specimen_codes[str_length(antweb_specimen_codes) > 3]

sum(str_length(antweb_specimen_codes) <= 3)
sum(grepl(" ",antweb_specimen_codes))

antweb_specimen_codes <- unique(antweb_specimen_codes)


test <- read.csv("../data/antweb_records_raw.csv",sep = "\t",header=T) #, sep = "\t",header=F)

colnames(test)

library(dplyr)

test <- rename(test, "recordID" = "code")
test$collectioncode
test$code

records <- read.csv("../data/antweb_records.csv")
