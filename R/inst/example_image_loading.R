# test loading images via imageR

library(imager)
library(ggplot2)
library(raster)
img <- load.image("../data/segments/INATRANDOM-210309_segment.png")
pixels <- img[img[,,,1] != 0]

pix <- as.pixset(img)
arr <- as.array(img)
mat <- as.data.frame(img)

plot(img)

colors <- unique(arr)

# Create an RGB image from the raster stack
img_df <- as.data.frame(img)

# plot a histogram of pixel values
ggplot(img_df,aes(value))+geom_histogram(bins=30)

# testing loading images as raster stacks
# doesn't work currently

# Use stack function to read in all bands
img_stack <- stack("../../experiments/odo_seg_analyzer/images/4155610_198_discrete.png")
