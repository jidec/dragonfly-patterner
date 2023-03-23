src("src/extractSimpleColorStats.R")
colors_row <- extractSimpleColorStats("E:/dragonfly-patterner/experiments/patternize_recolorize/wings_for_brown_heat")[1,2:20] * 255
# colors row just has the unique colors that you might want to plot a heat of
colors_row

pattern_dir <- "E:/dragonfly-patterner/experiments/patternize_recolorize/wings_for_brown_heat"
target_rgb <- c(47, 30, 21)

ref_img_name <- 'WRK-000002_fore_pattern'

plotPatternElementHeatmap <- function(pattern_dir, target_rgb, ref_img_name){
  
  library(patternize)
  library(stringr)
  
  # List with samples
  IDlist <- str_replace(list.files(pattern_dir),".png","")
  
  # make list with images
  extension <- '.png'
  imageList <- makeList(IDlist, 'image', pattern_dir, extension)
  
  # choose target image
  target <- imageList[[ref_img_name]]
  
  
  # BELOW - is the part that needs fixing
  # run alignment of color patterns
  rasterList_regRGB <- patRegRGB(imageList, target, target_rgb, resampleFactor = 5, colOffset= 0.15, 
                                 removebg = 100, plot = 'stack')
  # sum the colorpatterns
  summedRaster_regRGB <- sumRaster(rasterList_regRGB, IDlist, type = 'RGB')
  
  # plot heat
  colfunc <- c("black","lightblue","blue","green", "yellow","red")
  plotHeat(summedRaster_regRGB, IDlist, plotCartoon = FALSE, refShape = 'target', flipRaster = 'xy', imageList = imageList, cartoonID = 'BC0004', 
           cartoonFill = 'black', cartoonOrder = 'under', colpalette = colfunc)
}

