# messing around with imageR 

install.packages("imager")
install.packages("ggplot2")
library(imager)
library(ggplot2)

img <- load.image("../prelim_experiments/odo_seg_alignment/segs/4155610_198_masked_clust.jpg")
plot(img)
img_df <- as.data.frame(img)
plot(img_df)

ggplot(img_df,aes(value))+geom_histogram(bins=30)
