library(patternize)

# lists = read in all aligned images 

pcaOut <- patPCA(TotalList, popList, colList, symbolList = symbolList, plot = TRUE, plotType = 'points', plotChanges = TRUE, PCx = 1, PCy = 2, 
                 plotCartoon = TRUE, refShape = 'target', outline = outline_9472, colpalette = colfunc, 
                 crop = c(300,2800,300,1800),flipRaster = 'y', imageList = imageListWT, cartoonID = 'cross20_F1fBC1_wtm_9472', 
                 normalized = TRUE, cartoonFill = 'black', cartoonOrder = 'under', legendTitle = 'Predicted')