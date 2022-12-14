# plot inference statistics

# count number of each class
table(inferences$dorsal_lateral_bad)
table(inferences$conf_infers1)
table(inferences$conf_infers3)
table(inferences$conf_infers5)
table(inferences$conf_infers7)

# get percents of each class
infer_confs <- data.frame(rbind(prop.table(table(inferences$dorsal_lateral_bad)),
                                prop.table(table(inferences$conf_infers1)),
                                prop.table(table(inferences$conf_infers3)),
                                prop.table(table(inferences$conf_infers5)),
                                prop.table(table(inferences$conf_infers7))))
infer_confs$conf_cutoff <- c(0,1,3,5,7)

# add new cols
infer_confs$percent_kept = 1 - infer_confs$bad
infer_confs$percent_og_dorsal_kept <- infer_confs[,2]/infer_confs[1,2]
infer_confs$percent_og_lateral_kept <- infer_confs[,3]/infer_confs[1,3]

View(infer_confs)

library(ggplot2)
# plot percent of images
ggplot(infer_confs, aes(x=conf_cutoff)) +
  geom_line(aes(y = bad), color = "red") + 
  geom_line(aes(y = dorsal), color = "green") + 
  geom_line(aes(y = lateral), color = "blue") + 
  ylim(c(0,1)) + 
  xlab("Weight confidence cutoff") + ylab("Percent of images") 

# plot percent of images in each class kept
ggplot(infer_confs, aes(x=conf_cutoff)) +
  geom_line(aes(y = percent_og_dorsal_kept), color = "green") + 
  geom_line(aes(y = percent_og_lateral_kept), color = "blue") + 
  ylim(c(0,1)) +
  xlab("Weight confidence cutoff") + ylab("Percent of images in each class kept")

library(dplyr)
seg_infers <- filter(inferences,!is.na(bad_signifier))

# percent of bad segments
sum(seg_infers$bad_signifier == 1) / nrow(seg_infers)
