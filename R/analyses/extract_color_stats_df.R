source("src/extractSimpleColorStats.R")
dir <- "../data/segments"
img_locs <- paste0(dir,"/", setdiff(list.files(dir), list.dirs(recursive = FALSE, full.names = FALSE)))
img_locs <- img_locs[-1]
color_stats <- extractSimpleColorStats(img_locs)
mean_colors <- extractMeanColor(img_locs)
