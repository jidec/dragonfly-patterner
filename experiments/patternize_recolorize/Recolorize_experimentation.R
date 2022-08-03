
library(recolorize)

setwd("/Users/louiseppel/Documents/GitHub/dragonfly-patterner/experiments/patternize:recolorize/segments/example_clade")

#create a list of the files from target directory
file_list <- list.files(path="/Users/louiseppel/Documents/GitHub/dragonfly-patterner/experiments/patternize:recolorize/segments/example_clade")

viewChannels <- function(file_list) {
  for (i in 1:length(file_list)){
  # load image
  print(file_list[i])
  img <- readImage(file_list[i], 
                 resize = NULL, rotate = NULL)
  #plot channels
  layout(matrix(1:4, nrow = 1))
  par(mar = c(0, 0, 2, 0))
  plotImageArray(img, main = "RGB image")
  plotImageArray(img[ , , 1], main = "R channel")
  plotImageArray(img[ , , 2], main = "G channel")
  plotImageArray(img[ , , 3], main = "B channel")
  # if I can figure out how to add an alpha channel then the following line will include a mask
  # plotImageArray(img[ , , 4], main = "Alpha channel")
  }
}

# viewChannels(file_list)

# loop through a file of images and blur them
blurImages <- function(file_list, blur_function, amplitude, sharpness) {
  for (i in 1:length(file_list)) {
    print(file_list[i])
    img <- readImage(file_list[i], 
                     resize = NULL, rotate = NULL)
    blurred_img <- blurImage(img, blur_function,
                             amplitude, sharpness)
  }
}

# blurImages(file_list, blur_function = "blur_anisotropic", 
#                         amplitude = 10, sharpness = 0.2)

# recolorize images
recolorizeImages <- function(file_list, bins, method) {
  for (i in 1:length(file_list)) {
    print(file_list[i])
    img <- readImage(file_list[i], 
                     resize = NULL, rotate = NULL)
    # use function to determine number of bins
    recolorize(img = img, bins = 4, method = 'hist')
  }
}

# dealing with shine experimentation
library(recolorize)
img <- "/Users/louiseppel/Documents/GitHub/dragonfly-patterner/experiments/patternize:recolorize/segments/example_clade/INATRANDOM-23567781_segment.png"
s0 <- recolorize2(img, bins = 2, cutoff = 20, color_space = "sRGB")
# plot(s0)


