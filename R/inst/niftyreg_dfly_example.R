files <- list.files(path="E:/dragonfly-patterner/data/patterns/gomph_grouped_5000",full.names = TRUE)
files[1]
files[2]
img <- readPNG(files[1])
img2 <- readPNG(files[3])
display(img)
display(img2)

result <- niftyreg(img2, img1)
display(result$image)
result$image
dim(img2)
dim(img1)
img1 <- readPNG("E:/dragonfly-patterner/data/patterns/gomph_grouped_5000/INAT-133878409-2_pattern.png")
img2 <- readPNG("E:/dragonfly-patterner/data/patterns/gomph_grouped_5000/INAT-82301046-2_pattern.png")
