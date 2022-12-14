library(emdist)
library(imager)
imgs <- load.dir("E:/dragonfly-patterner/data/patterns/gomph_grouped_5000")

lightness_mats <- list()
for(i in 1:length(imgs)){
  r <- as.matrix(as.array(imgs[[i]])[,,,1])
  g <- as.matrix(as.array(imgs[[i]])[,,,2])
  b <- as.matrix(as.array(imgs[[i]])[,,,3])
  l <- (r + g + b) / 3
  lightness_mats <- append(lightness_mats,list(l))
  print(i)
}

lightness_mats[[1]]
names(lightness_mats) <- names(imgs)

emds <- matrix(nrow=length(lightness_mats),ncol=length(lightness_mats))
for(m in 1:length(lightness_mats)){
  for(m2 in 1:length(lightness_mats)){
    emds[m,m2] <- emd(lightness_mats[[m]],lightness_mats[[m2]])
  }
  print(m)
}

rownames(emds) <- names(imgs)
colnames(emds) <- names(imgs)
View(rowMeans(emds)[1:100])

