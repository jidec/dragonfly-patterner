###
# patternize registration/RGB analysis of ten Heliconius erato hydara
###

# List with samples
IDlist <- c('BC0004',
            'BC0049',
            'BC0050',
            'BC0071',
            'BC0077',
            'BC0079',
            'BC0082',
            'BC0125',
            'BC0129',
            'BC0366')

# make list with images
prepath <- 'images/Heliconius'
extension <- '-D.jpg'
imageList <- makeList(IDlist, 'image', prepath, extension)

# choose target image
target <- imageList[['BC0004']]

# run alignment of color patterns
RGB <- c(114,17,0) # red
rasterList_regRGB <- patRegRGB(imageList, target, RGB, resampleFactor = 5, colOffset= 0.15, 
                               removebg = 100, plot = 'stack')

# If you don't want to run the function, you can load the saved output rasterList
# save(rasterList_regRGB, file = 'output/Fig1_rasterList_regRGB.rda')
load('output/Fig1_rasterList_regRGB.rda')

# sum the colorpatterns
summedRaster_regRGB <- sumRaster(rasterList_regRGB, IDlist, type = 'RGB')

# plot heatmap
outline_BC0004 <- read.table('cartoon/BC0004_outline.txt', h= F)
lines_BC0004 <- list.files(path='cartoon', pattern='BC0004_vein', full.names = T)

# colfunc <- c("black","lightblue","blue","green", "yellow","red")
colfunc <- inferno(100)
plotHeat(summedRaster_regRGB, IDlist, plotCartoon = TRUE, refShape = 'target', outline = outline_BC0004, 
         lines = lines_BC0004, flipRaster = 'xy', imageList = imageList, cartoonID = 'BC0004', 
         cartoonFill = 'black', cartoonOrder = 'under', colpalette = colfunc)

##
# Compare landmark and registration
##

# give rasters same extent and resolution
rasterEx <- raster::extent(min(outline_BC0004[,1]),
                           max(outline_BC0004[,1]),
                           min(outline_BC0004[,2]),
                           max(outline_BC0004[,2]))

rRe <- raster::raster(nrow=200,ncol=200)
raster::extent(rRe) <- rasterEx

summedRaster_lanRGB2 <- raster::resample(summedRaster_lanRGB,rRe,datatype="INT1U", method='ngb')
summedRaster_regRGB2 <- raster::resample(summedRaster_regRGB,rRe,datatype="INT1U", method='ngb')

# orient rasters
summedRaster_lanRGB2 <- raster::flip(summedRaster_lanRGB2, 'y')
summedRaster_regRGB2 <- raster::flip(summedRaster_regRGB2, 'y')
summedRaster_regRGB2 <- raster::flip(summedRaster_regRGB2, 'x')

# subtract rasters
subtracted <- summedRaster_lanRGB2/length(IDlist) - summedRaster_regRGB2/length(IDlist)

# plot heatmap
colfunc <- c("blue","lightblue","black","pink","red")
plotHeat(subtracted, IDlist, plotCartoon = TRUE, refShape = 'target', outline = outline_BC0004, 
         lines = lines_BC0004, landList = landmarkList, adjustCoords = TRUE, imageList = imageList, 
         normalized = TRUE,cartoonID = 'BC0004', zlim=c(-1,1), colpalette= colfunc, cartoonFill = 'black', 
         cartoonOrder = 'under', legendTitle = 'Difference')
