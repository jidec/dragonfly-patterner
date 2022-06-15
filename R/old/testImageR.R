# messing around with imageR 

install.packages("imager")
install.packages("ggplot2")
library(imager)
library(ggplot2)
library(raster)
img <- load.image("../experiments/odo_seg_analyzer/images/4155610_198_discrete.png")
pixels <- img[img[,,,1] != 0]
pixels
pix <- as.pixset(img)
arr <- as.array(img)
mat <- as.data.frame(img)
220864/4
dim(arr) <- c(55216,4)
arr
colors <- unique(arr)

sum(arr == colors[4,])
for(c in colors){
  
}

# Use stack function to read in all bands
RGB_stack_HARV <- stack("../../experiments/odo_seg_analyzer/images/4155610_198_discrete.png")
# Create an RGB image from the raster stack
plotRGB(imageList[[1]], 
        r = 1, g = 2, b = 3)
plot(img)
img_df <- as.data.frame(img)
plot(img_df)

ggplot(img_df,aes(value))+geom_histogram(bins=30)
