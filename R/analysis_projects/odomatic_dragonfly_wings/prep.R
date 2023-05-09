
# we start by clustering wing segments to patterns using recolorize in R, which has a
# blurring function and hierarchical clustering method that works great on wing veins
source("src/recolorizeWingSegmentDir.R")
recolorizeWingSegmentDir(dir="D:/wing-color/data/segments","D:/wing-color/data/patterns",median_blur_n = 7,start_index = 6304,cluster_cutoff = 25) # missed 4530-33

# in Python, we group cluster the patterns in the data/patterns folder to data/patterns/grouped

# prep wing color data using the grouped patterns
# before running this, must manually load in SDMs in data folder via "Import Dataset"
source("src/prepWingData.R")
prepWingData(wing_img_loc = "D:/wing-color/data/patterns/grouped2",
             patpca_img_loc = "D:/wing-color/data/patterns/grouped2resized",
             tc1_1 = 0.5529, tc1_2 = 0.2470)

# returns a list of dataframes in list indices:
# 1. individual-level data
# 2. individual-level data trimmed to phy
# 3. phy trimmed to data
# 4. species-level data
# 5. species-level data trimmed to phy

# 6. species-sex level data
# 7. species-sex-forehind level data
