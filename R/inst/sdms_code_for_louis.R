
# load sdm data (paper: https://www.mdpi.com/1424-2818/14/7/575)
library(readr)
sdm_data <- read_csv("D:/wing-color/data/Nearctic_ModelResults_5KM.csv")

# takes awhile (up to hours) and is probably inefficient, but returns pairwise range overlaps
# save this data somewhere once it's complete! 
source("src/getSpRangeOverlapsFromSDMs.R")
range_ol <- getSpRangeOverlapsFromSDMs(sdm_data)

# stuff below could be useful but you probably dont need
source("src/getSpPairwiseMetricDiffs.R")
l_diffs <- getSpPairwiseMetricDiffs(wings_sp,"col_6_mean") # get pairwise differences in wing lightness 
overlap_ldiffs <- merge(range_ol, l_diffs, by=c("sp1","sp2")) # merge trait with range overlap

