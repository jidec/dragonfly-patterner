# some code to compare shared annotations

# read in annotations
annots_j <- read.csv("../annotations/Shared/1-20-22_Stylurus200/Jacob_imagant/session.csv")
annots_l <- read.csv("../annotations/Shared/1-20-22_Stylurus200/Louis_imageant/session1.csv")

# look at annotations
annots_j$dorsal_lateral
annots_l$dorsal_lateral

# get a vector of TRUE or FALSE describing which annotations are equal (equal annotations are TRUE) 
match <- annots_j$dorsal_lateral == annots_l$dorsal_lateral

#count the percent of annotations that the annotators described the same
sum(match) / 150 # 80% 

# count the number of each class 
table(annots_j$dorsal_lateral) # 80 bad, 29 dorsal, 41 lateral
table(annots_l$dorsal_lateral) # 74 bad, 37 dorsal, 39 lateral

