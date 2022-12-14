
# plot pixels in 3D
Heliconius_08 <- system.file("extdata", "Heliconius/Heliconius_B/Heliconius_08.jpeg", package="colordistance")
lower <- rep(0.8, 3)
upper <- rep(1, 3)
H8hist <- colordistance::getImageHist(Heliconius_08, bins=c(2, 2, 2), lower=lower, upper=upper)

colordistance::plotPixels(Heliconius_08, lower=NULL, upper=NULL)

# plot hist bins
images <- dir(system.file("extdata", "Heliconius/", package="colordistance"), full.names=TRUE)
histList <- colordistance::getHistList(images, lower=NULL, upper=NULL, bins=rep(2, 3), plotting=FALSE, pausing=FALSE)

# todo get earth movers dist between discretized patterns 
# find outliers by dist 

CDM <- colordistance::getColorDistanceMatrix(histList, method="emd", plotting=FALSE)
print(CDM)
colordistance::heatmapColorDistance(CDM)
View(CDM)
images <- list.files("E:/dragonfly-patterner/data/patterns/gomph_grouped_5000",full.names = TRUE)
images <- images[1:10]
histList <- colordistance::getHistList(images, lower=lower, upper=upper, bins=rep(2, 3), plotting=FALSE, pausing=FALSE)

CDM[is.na(CDM)] <- 0

View(rowMeans(CDM))
