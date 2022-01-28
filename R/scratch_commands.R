source("getGenusList.R")
source("mergeAnnotations.R")
source("moveSelectImages.R") # also contains moveAnnotationClassImages function

mergeAnnotations()

moveSelectImages(num_images=150, from="../downloaders/helpers/genus_image_records/iNat_images-Stylurus-raw_images",
                 to = "../data/all_images")

moveSelectImages(num_images = 150, from="../data/all_images",
                 to = "../annotations/Jacob/")

# move shared annotation class images to train set
moveAnnotationClassImages(class_name = "0", from = "../data/all_images",
                          to = "../experiments/odo_view_classifier/odo_view_data")

# move shared annotation class images to train set
moveAnnotationClassImages(class_name = "dorsal", from = "../data/all_images",
                          to = "../experiments/odo_view_classifier/odo_view_data")


# move shared annotation class images to train set
moveAnnotationClassImages(class_name = "lateral", from = "../data/all_images",
                          to = "../experiments/odo_view_classifier/odo_view_data")
