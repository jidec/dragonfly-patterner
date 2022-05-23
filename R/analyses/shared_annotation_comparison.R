# some code to compare shared annotations

# read in annotations
annots_j <- read.csv("../trainset_tasks/Shared/5-10-22_Classes_150/jacob_session.csv")
annots_l <- read.csv("../trainset_tasks/Shared/5-10-22_Classes_150/louis_session.csv")
annots_a <- read.csv("../trainset_tasks/Shared/5-10-22_Classes_150/ana_session.csv.csv")

# look at annotations
annots_j$class
annots_l$class
annots_a$class

# get a vector of TRUE or FALSE describing which annotations are equal (equal annotations are TRUE) 
match <- (annots_a$class == annots_j$class)
sum(match)
annots_j$file[match == FALSE]

#count the percent of annotations that the annotators described the same
sum(match) / 150 # 80% 

# count the number of each class 
table(annots_j$class) # 80 bad, 29 dorsal, 41 lateral
table(annots_l$class) # 74 bad, 37 dorsal, 39 lateral
table(annots_a$class)


annots <- read.csv("../data/annotations.csv")
table(annots$dorsal_lateral_dorsolateral_bad)
