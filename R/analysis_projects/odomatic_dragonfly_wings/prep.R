
# we start by clustering wing segments to patterns using recolorize in R, which has a
# blurring function and hierarchical clustering method that works great on wing veins
source("src/recolorizeWingSegmentDir.R")
recolorizeWingSegmentDir(dir="D:/wing-color/data/segments","D;/wing-color/data/patterns")

# in Python, we group cluster the patterns in the data/patterns folder to data/patterns/grouped

# prep wing color data using the grouped patterns
# before running this, must manually load in SDMs in data folder via "Import Dataset"
source("src/prepWingData.R")
data_tuple <- prepWingData(wing_img_loc = "D:/wing-color/data/patterns/grouped",
                           patpca_img_loc = "D:/wing-color/data/patterns/grouped/size_normalized")
#saveRDS(data_tuple,"wing_data_tuple.rds")
#data_tuple <- readRDS("saved/processed/wing_data_tuple.rds")

# returns a list of dataframes in list indices:
# 1. individual-level data
# 2. individual-level data trimmed to phy
# 3. phy trimmed to data
# 4. species-level data
# 5. species-level data trimmed to phy

# 6. species-sex level data
# 7. species-sex-forehind level data

wings <- data_tuple[[1]]
wings_phy_trim <- data_tuple[[2]]
wings_phy <- data_tuple[[3]]
phy <- wings_phy

wings_sp <- data_tuple[[4]]
wings_sp_phy_trim <- data_tuple[[5]]

wings_sp_sex <- data_tuple[[6]]
wings_sp_sex_type <- data_tuple[[7]]
