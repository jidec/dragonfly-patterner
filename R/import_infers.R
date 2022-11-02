# contains reading in inferences
# extracting and plotting confidence
# manually specifying accuracy 
# quick fun to plot images using id 
# counting the number of bads

new_inferences <- read.csv("E:/dragonfly-patterner/data/inferences.csv")
inferences <- read.csv("E:/dragonfly-patterner/data/inferences.csv")

table(inferences$dorsal_lateral_bad)
table(inferences$conf_infers1)
table(inferences$conf_infers3)
table(inferences$conf_infers5)
table(inferences$conf_infers7)

prop.table(table(inferences$dorsal_lateral_bad))
prop.table(table(inferences$conf_infers1))
prop.table(table(inferences$conf_infers3))
prop.table(table(inferences$conf_infers5))
prop.table(table(inferences$conf_infers7))

infer_confs <- data.frame(rbind(prop.table(table(inferences$dorsal_lateral_bad)),
      prop.table(table(inferences$conf_infers1)),
      prop.table(table(inferences$conf_infers3)),
      prop.table(table(inferences$conf_infers5)),
      prop.table(table(inferences$conf_infers7))))

infer_confs$percent_kept = 1 - infer_confs$bad
infer_confs$percent_dorsal_kept = 1 - infer_confs$bad - infer_confs$lateral


infer_confs$conf_cutoff <- c(0,1,3,5,7)

library(ggplot2)
ggplot(infer_confs, aes(y=percent_kept,x=conf_cutoff)) +
  geom_line(color="red") + xlab("Weight confidence cutoff") + ylab("Percent of all dorsals and laterals kept")
ggplot(infer_confs, aes(y=percent_dorsal_kept,x=conf_cutoff)) +
  geom_line(color="red")

library(dplyr)
inferences <- filter(inferences,is.na(bad_signifier))
inferences$true_class <- rep("",nrow(inferences))

n <- 25
library(imager)
for(s in 1:n){
  row_index <- sample(nrow(inferences),1)
  print(row_index)
  row <- inferences[row_index, ]
  print(row)
  id <- row$imageID
  print(id)
  img <- load.image(paste0('E:/dragonfly-patterner/data/all_images/',id,".jpg"))
  plot(img)
  class <- readline()
  inferences[row_index,]$true_class <- class
}

library(imager)
fpath <- system.file(paste0('E:/dragonfly-patterner/data/all_images/',sample_row$imageID,".jpg"),package='imager')
img <- load.image(paste0('E:/dragonfly-patterner/data/all_images/',sample_row$imageID,".jpg"))
plot(img)
sample_row$conf_infers3

var <- readline()

#inferences$true_class[inferences$true_class == "ba"] <- "bad" 
library(dplyr)

test <- filter(inferences,true_class != "")
test$true0 <- (test$dorsal_lateral_bad == test$true_class) | test$dorsal_lateral_bad == "bad"
test$true1 <- (test$conf_infers1 == test$true_class) | test$conf_infers1 == "bad"
test$true3 <- (test$conf_infers3 == test$true_class) | test$conf_infers3 == "bad"
test$true5 <- (test$conf_infers5 == test$true_class) | test$conf_infers5 == "bad"
test$true7 <- (test$conf_infers7 == test$true_class) | test$conf_infers7 == "bad"

table(test$true0)
table(test$true3)
table(test$true5)

test$id <- test$imageID

test$true_class
ids <- filter(test,true7 == FALSE)$imageID
ids
visualizeID(ids[4])
ids[1]
ids <- filter(test,true5 == FALSE)

ex <- filter(test, true5 == FALSE )
table(test$true7)

test$conf_infers1
(test$conf_infers1 == test$true_class)

test$true_bad0 <- test$conf_infers1 == test$true_class | test$conf_infers1 == "bad"

genera
id <- ex$imageID[4]

visualizeID <- function(id){
  library(imager)
  img <- load.image(paste0('E:/dragonfly-patterner/data/all_images/',id,".jpg"))
  return(plot(img))
}
visualizeID("INAT-49781305-4")

id <- "INAT-49781305-4"
img <- load.image(paste0('E:/dragonfly-patterner/data/all_images/',id,".jpg"))
plot(img)

#counting bads
inferences$imageID <- str_split_fixed(inferences$imageID, "-", n =3)[,2]
test <- merge(bound_records,filter(inferences, bad_signifier == 0), by="imageID")
test <- filter(test, bad_signifier == 0)
test <- duplicated(test)
duplicated(test)
