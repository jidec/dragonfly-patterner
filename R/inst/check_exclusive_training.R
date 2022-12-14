source("src/getMisplacedIDsFromFolder.R")
library(stringr)
training <- getIDsFromFolder("E:/dragonfly-patterner/data/other/train_masks")
rob1 <- getIDsFromFolder("E:/dragonfly-patterner/trainset_tasks/Rob/11-11-22_RobLatSegs200")
rob2 <- getIDsFromFolder("E:/dragonfly-patterner/trainset_tasks/Rob/11-16-22_DorsalSegs200")

both <- c(training,rob2)
both <- str_remove(both,".jpg")
length(both)
length(unique(both))

v <- c(TRUE,FALSE,'')
write.csv(v,"v.csv")
getwd()
