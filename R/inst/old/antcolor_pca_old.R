ants <- read.csv("D:/ant-patterner/specimen_colors_unadj.csv")
pca <- prcomp(ants[,c(3:5)], center = T,scale = T)
summary(pca)

library(devtools)
install_github("vqv/ggbiplot")

library(ggbiplot)

ggbiplot(pca)
pca$rotation

ants$pca1r <- pca$rotation[1] * ants$r
ants$pca1g <- pca$rotation[2] * ants$g
ants$pca1b <- pca$rotation[3] * ants$b
i <- 3333
plotRGBColor(c(ants$pca1r[i],ants$pca1g[i],ants$pca1b[i]))
plotRGBColor(c(ants$r[i],ants$g[i],ants$b[i]))
ants$pc1[i]
ants$pc2[i]

ants$pc1 <- (pca$rotation[1] * ants$r) + (pca$rotation[2] * ants$g) + (pca$rotation[3] * ants$b)
ants$pc2 <- (pca$rotation[4] * ants$r) + (pca$rotation[5] * ants$g) + (pca$rotation[6] * ants$b)


library(grDevices)
plotRGBColor <- function(rgbc) {
    # todo - finish allowing matrix of multiple rgbs
    #rgb <- rbind(rgb, rgb2)
    #rgb[,1], rgb[,2]
    col <- rgb(rgbc[1], rgbc[2], rgbc[3], maxColorValue = 1)
    return(plot(c(1), col = col, pch = 15, cex = 40, axes = FALSE, ylab='',xlab=''))
}

plotRGBColor(ants$pc)

pca$rotation

rot <- c(pca$rotation[1],pca$rotation[2],pca$rotation[3])
rot <- c(pca$rotation[4],pca$rotation[5],pca$rotation[6])
color <- c(125,125,125)

modColorByPC <- function(c,rot,i){
    c <- c + (i * rot)
    return(c)
}

for(i in 1:5){
    color <- modColorByPC(color,rot,25)
    plotRGBColor(color)
}

