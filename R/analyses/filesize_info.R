# infer some filesize info

dir <- "../data/segments/masks/train_masks"
dir <- "../data/all_images"

files <- paste0(dir, "/",list.files(dir))

mean_size_mb <- mean(file.size(files)) / 1000000

# get the approx size of 550000 images of type in gb
(mean_size_mb * 550000) / 1000

# results - masks should be 9 gb, raw images should be 250 gb 

mean_size_mb * 10000

# calculate number of obs to download given more than 1 image per obs
10000/ mean(data$numImages)

mean(data$numImages)

500 * mean(data$numImages)
