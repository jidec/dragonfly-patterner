
# plot pixels in 3D
Heliconius_08 <- system.file("extdata", "Heliconius/Heliconius_B/Heliconius_08.jpeg", package="colordistance")
colordistance::plotPixels(Heliconius_08, lower=NULL, upper=NULL)

# plot hist bins
images <- dir(system.file("extdata", "Heliconius/", package="colordistance"), full.names=TRUE)
histList <- colordistance::getHistList(images, lower=lower, upper=upper, bins=rep(2, 3), plotting=FALSE, pausing=FALSE)