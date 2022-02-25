source("src/getGenusList.R")
source("src/mergeAnnotations.R")
source("src/moveSelectImages.R") # also contains moveAnnotationClassImages function
source("src/smallHelpers.R")

mergeAnnotations()
annotations <- read.csv("../data/annotations.csv")
annotations <- annotations[!duplicated(annotations$file),]

moveSelectImages(num_images=150, from="../downloaders/helpers/genus_image_records/iNat_images-Stylurus-raw_images",
                 to = "../data/all_images")

moveSelectImages(num_images = 150, from="../data/all_images",
                 to = "../annotations/Jacob/")

# move shared annotation class images to train set
moveAnnotationClassImages(class_name = "0", from = "../data/all_images",
                          to = "../experiments/odo_view_classifier/odo_view_data",
                          split_test_train = TRUE, ntest = 10)

# move shared annotation class images to train set
moveAnnotationClassImages(class_name = "dorsal", from = "../data/all_images",
                          to = "../experiments/odo_view_classifier/odo_view_data",
                          split_test_train = TRUE, ntest = 4)

# move shared annotation class images to train set
moveAnnotationClassImages(class_name = "lateral", from = "../data/all_images",
                          to = "../experiments/odo_view_classifier/odo_view_data",
                          split_test_train = TRUE, ntest = 4)

# move exclusive images for segementation
moved <- moveSelectImages(num_images = 60, from="../data/all_images",
                 to = "../annotations/Jacob/2-4-22_60Segments")
moveSelectImages(num_images = 60, from="../data/all_images",
                          to = "../annotations/Louis/2-4-22_60Segments",excl_names = moved)

# move images that have not been annotated
exclude <- annotations$file
moveSelectImages(num_images = 150, from="../data/all_images",
                 to = "../annotations/Jacob/2-7-22_Classes150",excl_names = exclude)

# move exclusive dorsal and lateral images for segmentation
moved <- moveSelectImages(num_images = 30, class_name = "dorsal", from="../data/all_images",
                          to = "../annotations/Jacob/2-4-22_60Segments")
moveSelectImages(num_images = 30, class_name = "dorsal", from ="../data/all_images",
                 to = "../annotations/Louis/2-4-22_60Segments",excl_names = moved)

moved <- moveSelectImages(num_images = 30, class_name = "lateral", from="../data/all_images",
                          to = "../annotations/Louis/2-4-22_60Segments")
moveSelectImages(num_images = 30, class_name = "lateral", from ="../data/all_images",
                 to = "../annotations/Jacob/2-4-22_60Segments",excl_names = moved)

# check if dirs contain exclusive files
findNDuplicates("../annotations/Jacob/2-4-22_60Segments","../annotations/Louis/2-4-22_60Segments")

# remove images from annotation sets once annotations are done
deleteImagesFromDir("../annotations/Jacob/2-4-22_60Segments")

# move masks to data/segments/masks directory
moveMasks(from="../annotations/Jacob/2-4-22_60Segments")

# move masks to training and val directories
moveSegmentationMaskImages(ntest=7)

# open file browser to a directory
utils::browseURL("../annotations/Jacob/2-4-22_60Segments")
