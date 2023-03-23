library(colorEvoHelpers)

## QUESTION - what colors come out of optimized clustering? 
# plot pigment cluster colors
row <- wings[1,]
# three are notably darker, and looking at images correspond to visibly pigmented areas
plotRGB(c(row$col_1_r,row$col_1_g,row$col_1_b),max=1)
plotRGB(c(row$col_2_r,row$col_2_g,row$col_2_b),max=1)
plotRGB(c(row$col_6_r,row$col_6_g,row$col_6_b),max=1)
# other 3 are different shades of white and tan, shades of unpigmented wings
plotRGB(c(row$col_3_r,row$col_3_g,row$col_3_b),max=1)
plotRGB(c(row$col_4_r,row$col_4_g,row$col_4_b),max=1)
plotRGB(c(row$col_5_r,row$col_5_g,row$col_5_b),max=1)

## QUESTION - how are individuals spatially distributed?
# somewhat biased towards populated east coast regions, but not too bad 
plotLatLonHeatmap(wings)

## QUESTION - what 
runPlotBindPCA(as.data.frame(wings_sp_sex),c("black","brown","yellow"),colour_column="Sex")
runPlotBindPCA(wings,c("black","brown","yellow"),colour_column = "Sex")


typeof(wings_sp_sex$black)
typeof(wings_sp_sex$brown)
typeof(wings_sp_sex$yellow)

describeCols(wings,c("brown","black","yellow","PC1_foreall","PC2_foreall","PC1_hindall","PC2_hindall"))

# describe and plot numerics
describeCols(wings,c("col_1_prop","col_2_prop","col_6_prop"),scatter_xyz_names = c("col_1_prop","col_2_prop","col_6_prop")) # add PCs
# describe and plot factors
describeCols(wings,c("Sex","Species","Genus","Subfamily"),factors = TRUE) 

# view in text associations between pertinent variables
getDfAssoc(select(wings,c("col_1_prop","col_2_prop","col_6_prop","Genus","Family","Suborder","Sex","wing_type")))

pig_prop_pca <- runPlotBindPCA(wings,c("black","brown","yellow"))
pca_list[[2]]
pca_list[[3]]

hist(wings$col_6_prop,main="Few have a large percent of black pigment, but many have a little bit (pterostigma!)",
     xlab="Percent dark pigment in wing", breaks=200)
hist(wings$col_2_prop,main="Fewer have ANY brown pigment, but those that do tend to have more", breaks=200)
hist(wings$col_1_prop,main="Fewer still have ANY yellow pigment, but those that do tend to have more", breaks=200)

uniq_sp <- unique(wings$species)
out <- c()

# for each unique species
for(s in uniq_sp){
  species_has_m_and_f <- nrow(dplyr::filter(wings,species==s & Sex=="M")) > 0 & nrow(dplyr::filter(wings,species==s & Sex=="F")) > 0
  out <- c(out,species_has_m_and_f)
}
sum(out)
uniq_sp

# examine the combos
wings$col_bby <- wings$col_1_prop + wings$col_2_prop + wings$col_6_prop
wings$col_bb <- wings$col_2_prop + wings$col_6_prop
wings$col_bry <- wings$col_1_prop + wings$col_2_prop
wings$col_bly <- wings$col_1_prop + wings$col_6_prop
hist(wings$col_bby,main="Peak is AFTER the start for sum of all pigments", breaks=200)
hist(wings$col_bb,main="Peak is NOT AFTER the start for sum of all pigments", breaks=200)
hist(wings$col_bry,main="Yellow causes the peak to be after the start - many have some yellow", breaks=200)
hist(wings$col_bly,main="Cont", breaks=200)

wings$y_over_bby <- wings$col_1_prop / wings$col_bby
wings$br_over_bby <- wings$col_2_prop / wings$col_bby
wings$bl_over_bby <- wings$col_6_prop / wings$col_bby
hist(wings$y_over_bby,main="Yellow over total - some zero meaning no yellow but most have some", breaks=200)
hist(wings$br_over_bby,main="Brown over total - many have 1 meaning brown but no black or yellow", breaks=200)
hist(wings$bl_over_bby,main="Black over total - ALMOST ALL have zero meaning no black", breaks=200)

# brown is the most prevalent color by mean
mean(wings$col_6_prop)
mean(wings$col_2_prop)
mean(wings$col_1_prop)

# yellow and brown are tied as the prevalent by occurence, black far behind
wings$has_br <- wings$col_2_prop > 0
wings$has_bl <- wings$col_6_prop > 0
wings$has_y <- wings$col_1_prop > 0
sum(wings$has_y)
sum(wings$has_bl)
sum(wings$has_br)
