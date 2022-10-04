source("src/countContains.R")

#countContains("E:/dragonfly

# check if catalogNumber actually has .0s etc 
recs <- read.csv("../data/records.csv")
recs$catalogNumber

# check to make sure images are exclusive 
(tr_meta_segs)
#investigate train_metadata.csv
tr_meta <- read.csv("../data/train_metadata.csv")
View(tr_meta)

head(tr_meta)
sum(tr_meta$has_segment_x.1 == 1)

# get classed
seg_class <- tr_meta[tr_meta$class != -1,]
nrow(seg_class)

# get segged
seg_class <- tr_meta[tr_meta$has_segment == 1,]
nrow(seg_class)

# get overlap
seg_class <- seg_class[seg_class$class != -1,]
nrow(seg_class)

table(tr_meta$class)
table(tr_meta$has_segment)

View(tr_meta)
duplicated(tr_meta$imageID)
tr_meta$imageID == tr_meta$imageID[513]
sum(duplicated(tr_meta$imageID))

sum(duplicated(tr_meta$imageID))


tr_meta$file[tr_meta$imageID == tr_meta$imageID[513]]
tr_meta$has_segment_x == tr_meta$has_segment_y

# check if there are any duplicates where 1 is a class and 1 is a seg
segs_nondup <- tr_meta_segs$imageID[!duplicated(tr_meta_segs$imageID)]
nonsegs_nondup <- tr_meta_nonsegs$imageID[!duplicated(tr_meta_nonsegs$imageID)]
sum(duplicated(c(segs_nondup,nonsegs_nondup)))

# investigate train_metadata.csv
cl_meta <- read.csv("../data/class_metadata.csv")
s_meta <- read.csv("../data/seg_metadata.csv")
sum(duplicated(s_meta$imageID))
View(s_meta)

outersect <- function(x, y) {
  sort(c(setdiff(x, y),
         setdiff(y, x)))
}

library(stringr)
train_img <- list.files("E:/dragonfly-patterner/data/other/training_dirs/new_segmenter/train/image")
train_mask <- list.files("E:/dragonfly-patterner/data/other/training_dirs/new_segmenter/train/mask")
train_mask <- str_replace_all(train_mask,"_mask","")
outer <- setdiff(train_img,train_mask)
outer <- paste0("E:/dragonfly-patterner/data/other/training_dirs/new_segmenter/train/image/",outer)
unlink(outer)

test_img <- list.files("E:/dragonfly-patterner/data/other/training_dirs/new_segmenter/test/image")
test_mask <- list.files("E:/dragonfly-patterner/data/other/training_dirs/new_segmenter/test/mask")
test_mask <- str_replace_all(test_mask,"_mask","")
outer <- setdiff(test_img,test_mask)
outer <- paste0("E:/dragonfly-patterner/data/other/training_dirs/new_segmenter/test/image/",outer)
unlink(outer)

t <- colors[colors$recordID == "antweb1008009",] 
unlist(t)[2:4] * 255
as.numeric(unlist(t)[2:4]) * 255
