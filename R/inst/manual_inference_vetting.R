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

#inferences$true_class[inferences$true_class == "ba"] <- "bad" 
library(dplyr)

test <- filter(inferences,true_class != "")
test$id <- test$imageID
test$true0 <- (test$dorsal_lateral_bad == test$true_class) | test$dorsal_lateral_bad == "bad"
test$true1 <- (test$conf_infers1 == test$true_class) | test$conf_infers1 == "bad"
test$true3 <- (test$conf_infers3 == test$true_class) | test$conf_infers3 == "bad"
test$true5 <- (test$conf_infers5 == test$true_class) | test$conf_infers5 == "bad"
test$true7 <- (test$conf_infers7 == test$true_class) | test$conf_infers7 == "bad"
table(test$true0)
table(test$true3)
table(test$true5)

# get false ids
f_ids <- filter(test,true7 == FALSE)$imageID
visualizeID(f_ids[4])