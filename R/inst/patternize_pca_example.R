library(patternize)
library(RNiftyReg)
library(jpeg)
library(png)
library(mmand)

source <- readNifti(system.file("extdata", "epi_t2.nii.gz", package="RNiftyReg"))
target <- readNifti(system.file("extdata", "flash_t1.nii.gz", package="RNiftyReg"))
display(result)

result <- niftyreg(source, target)

display(result$image)
result$image
library(imager)
plot(as.cimg(result$image))
pcaOut <- patPCA(TotalList, popList, colList, symbolList = symbolList, plot = TRUE, plotType = 'points', plotChanges = TRUE, PCx = 1, PCy = 2, 
                 plotCartoon = TRUE, refShape = 'target', outline = outline_9472, colpalette = colfunc, 
                 crop = c(300,2800,300,1800),flipRaster = 'y', imageList = imageListWT, cartoonID = 'cross20_F1fBC1_wtm_9472', 
                 normalized = TRUE, cartoonFill = 'black', cartoonOrder = 'under', legendTitle = 'Predicted')

files <- list.files("E:/dragonfly-patterner/data/patterns/gomph_grouped_5000",full.names = TRUE)
img1 <- "E:/dragonfly-patterner/data/patterns/gomph_grouped_5000/INAT-133878409-2_pattern.png"
img2 <- "E:/dragonfly-patterner/data/patterns/gomph_grouped_5000/INAT-82301046-2_pattern.png"
library(patternize)
rasterList_regW <- patRegRGB(img1, img2)

data(rasterList_lanRGB)
View(rasterList_lanRGB)
pop1 <- c('BC0077','BC0071')
pop2 <- c('BC0050','BC0049','BC0004')
popList <- list(pop1, pop2)
colList <- c("red", "blue")

pcaOut <- patPCA(rasterList_lanRGB, popList, colList, plot = TRUE)

plot(rasterList_lanRGB[[1]])
table(rasterList_lanRGB[[1]]@data@values)

img2 <- readPNG("E:/dragonfly-patterner/data/patterns/gomph_grouped_5000/INAT-82301046-2_pattern.png")

img2[1,1,] == 
  
# reshape to pixels

# find pixels equaling cluster and set them to 1
# set all others to 0 
# convert to rasterstack
library(imager)
arr <- as.array(load.image("E:/dragonfly-patterner/data/patterns/gomph_grouped_5000/INAT-82301046-2_pattern.png"))
dim(arr) <- c(dim(arr)[1] * dim(arr)[2],4)
unique(arr)

getClustMaskForPatternize <- function(img_loc,clustval1){
  library(imager)
  img <- load.image(img_loc)
  arr <- as.array(img)
  og_dim <- dim(arr)
  dim(arr) <- c(dim(arr)[1] * dim(arr)[2],4)
  clust_mask <- abs(arr[,1] - clustval1) < 0.0001
  clust_mask[clust_mask == TRUE] <- 1
  clust_mask[clust_mask == FALSE] <- NA
  dim(clust_mask) <- c(og_dim[1],og_dim[2])
  library(raster)
  clust_mask <- raster(clust_mask)
  return(clust_mask)
}

getClustMaskForPatternize2 <- function(img_loc,clustval1,target_loc){
  library(imager)
  library(png)
  img <- readPNG(img_loc)
  target <- readPNG(target_loc)
  img <- niftyreg(img, target)$image
  
  #img <- load.image(img_loc)
  arr <- as.array(img)
  og_dim <- dim(arr)
  dim(arr) <- c(dim(arr)[1] * dim(arr)[2],4)
  clust_mask <- abs(arr[,1] - clustval1) < 0.0001
  clust_mask[clust_mask == TRUE] <- 1
  clust_mask[clust_mask == FALSE] <- NA
  dim(clust_mask) <- c(og_dim[1],og_dim[2])
  library(raster)
  clust_mask <- raster(clust_mask)
  return(clust_mask)
}

mask <- getClustMaskForPatternize2("E:/dragonfly-patterner/data/patterns/gomph_grouped_5000/INAT-82301046-2_pattern.png",0.7254902,"E:/dragonfly-patterner/data/patterns/gomph_grouped_5000/INAT-124953002-2_pattern.png")
mask2 <- getClustMaskForPatternize("E:/dragonfly-patterner/data/patterns/gomph_grouped_5000/INAT-124953002-2_pattern.png",0.7254902)
mask2 <- t(mask2)
raster_list <- list(mask,mask2)

patPCA(raster_list)

pop1 <- c('BC0077','BC0071')
pop2 <- c('BC0050','BC0049','BC0004')
popList <- list(pop1, pop2)
colList <- c("red", "blue")

pcaOut <- patPCA(raster_list, popList, colList, plot = TRUE)
View(result$image)
