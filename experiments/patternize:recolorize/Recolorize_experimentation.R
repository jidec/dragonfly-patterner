
library(recolorize)

setwd("/Users/louiseppel/Desktop/Dragonfiles/Argia_inculta-Argia_oculata")

#create a list of the files from your target directory
file_list <- list.files(path="/Users/louiseppel/Desktop/Dragonfiles/Argia_inculta-Argia_oculata")

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
  
  blurred_img <- blurImage(img, blur_function = "blur_anisotropic",
                         amplitude = 10, sharpness = 0.2)
  
  #what do the following two lines of code do?
  layout(matrix(1:4, nrow = 1))
  par(mar = c(0, 0, 2, 0))

  recolorize_defaults <- recolorize(img = blurred_img, bins = 4, method = 'hist')
}
