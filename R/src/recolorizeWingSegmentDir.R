
# define a custom recolorize function
recolorizeWingSegmentDir <- function(dir,write_dir,start_index=1,median_blur_n=5,cluster_cutoff=25) {

  library(recolorize)
  wings <- list.files(path=dir)#,full.names=TRUE)
  wings <- wings[start_index:length(wings)]
  counter <- 0

  #w <- "MLM-000009.tif_fore.png"
  #w <- wings[7]
  for(w in wings){
    # blur image
    rc0 <- blurImage(readImage(paste0(dir,"/",w), resize = 0.5),
                     blur_function = "medianblur", median_blur_n, plotting = FALSE)

    # potential loop to fix errors where image is blurred such that no clusters are captured - this would throw a recolorize error
    # blur_sub <- median_blur_n
    # stop <- FALSE
    # an.error.occurred <- FALSE
    # while(!stop){
    #     tryCatch( {rc1 <- recolorize2(rc0, bins = c(3,3,3), cutoff = cluster_cutoff, plotting = FALSE)}
    #               , error = function(e) {an.error.occurred <<- TRUE})
    #     if(an.error.occurred){
    #         stop <- FALSE
    #
    #         blur_sub <- blur_sub - 1
    #         rc0 <- blurImage(readImage(paste0(dir,"/",w), resize = 0.5),
    #                          blur_function = "medianblur", blur_sub, plotting = TRUE)
    #     }
    #     else{
    #         stop <- TRUE
    #     }
    # }

    errored <- FALSE
    tryCatch( {rc1 <- recolorize2(rc0, bins = c(3,3,3), cutoff = cluster_cutoff, plotting = FALSE)}
                             , error = function(e) {errored <<- TRUE})

    if(errored){
        print("Only 1 cluster found, skipping image")
    }
    else{
        # drop minor colors
        # rc2 <- thresholdRecolor(rc1, plotting = FALSE)
        # and recluster again
        if(length(rc1$sizes) > 1){
            rc1 <- recluster(rc1, cutoff = cluster_cutoff, plot_final = FALSE, plot_hclust = FALSE)
        }

        recolorize_to_png(rc1,paste0(write_dir,"/",w))
    }
    print(counter + start_index)
    counter <- counter + 1
  }
}

#?recolorize::recolorize_to_png
#plotRGBColor(c(row$col_6_r,row$col_6_g,row$col_6_b),max=1)

#2651
#recolorizeWingSegmentDir(dir,write_dir,4530)
# library(recolorize)
# rc0 <- blurImage(readImage("D:/wing-color/data/segments/MLM-000107_fore_segment.png", resize = 0.5),
#                  blur_function = "medianblur", n = 3, plotting = TRUE)
# rc0 <- blurImage(readImage("D:/wing-color/data/segments/MLM-000051_fore_segment.png", resize = 0.5),
#                  blur_function = "medianblur", n = 5, plotting = TRUE)
# rc1 <- recolorize2(rc0, bins = c(3,3,3), cutoff = 25, plotting = TRUE)
# rc1 <- recolorize2(rc0, bins = c(3,3,3), cutoff = 50, plotting = TRUE)
#
# ?recolorize2

# img <- readImage("D:/wing-color/data/patterns/MLM-000218_hind_segment.png")
# img <- as.array(img) # convert to arr
# dim(img) <- c(dim(img)[1] * dim(img)[2],4) # reshape to pixels
#
# # get unique colors
# colors <- unique(arr)
# colors
# # remove black background pixels
# arr <- arr[(arr != c(0,0,0,0))[,1],]
#
# # get name (id) from path
# name <- strsplit(path, "/", fixed = TRUE)[[1]]
# name <- name[length(name)]
# name <- strsplit(name, "_", fixed = TRUE)[[1]]
# name <- name[1]
#
# # get unique colors
# colors <- unique(arr)
