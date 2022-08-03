source("src/countContains.R")
contains = "_h"
direct_dir = "E:/ant-patterner/data/all_images"
str_subset(contains, list.files(direct_dir))


countContains("_h","E:/ant-patterner/data/all_images")
test <- list.files(direct_dir)
length(test)
length(str_subset(test,contains))


length(unique(out[,1]))
out

files <- list.files("E:/ant-patterner/data/all_images", pattern = ".jpg", recursive = TRUE)


t <- str_split_fixed(files,"/",n=2)[,2]
View(t)
t <- str_split_fixed(t,"_",n=3)
v <- t3[,1]
length(unique(v))
v2 <- str_subset(files,"_h")
length(v2)
View(t)

names <- paste(t[,1],t[,2])
length(unique(str_subset(paste(t[,1],t[,2])," h")))

# 53588 in new data 
names
files