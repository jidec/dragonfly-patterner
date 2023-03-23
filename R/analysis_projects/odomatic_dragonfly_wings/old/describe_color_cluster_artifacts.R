wings <- wings[!duplicated(as.list(wings))]

library(ggplot2)

# hists of all pigment percents
hist(wings$col_6_prop,main="Few have a large percent of black pigment, but many have a little bit (pterostigma!)",
     xlab="Percent dark pigment in wing", breaks=200)
hist(wings$col_2_prop,main="Fewer have ANY brown pigment, but those that do tend to have more", breaks=200)
hist(wings$col_1_prop,main="Fewer still have ANY yellow pigment, but those that do tend to have more", breaks=200)

# cor between each pigment percent
cor.test(wings$col_6_prop,wings$col_2_prop) # black is uncorrelated with brown
# reasons:
# 1. more black could mean less brown (-cor)
# 2. but ALSO brown and black could tend to occur together (we know this happens phylogenetically) (+cor)

cor.test(wings$col_1_prop,wings$col_2_prop) # yellow is correlated with brown

# examine how the pigment props cluster
plot(wings$col_6_prop,wings$col_2_prop)

pca <- prcomp(cbind(wings$col_6_prop,wings$col_2_prop,wings$col_1_prop))
summary(pca)
pca$rotation
library(ggfortify)
wings <- cbind(wings,pca$x)

# PC1 - less brown
# PC2 - more black, somewhat more yellow, slightly less brown
# PC3 - more yellow, some less black and brown

ggplot(wings,aes(PC1,PC2)) + 
  geom_point(aes(colour=mean_lightness)) + ggtitle("PCA")
# shows that PC2 represents black
# PC1 x-axis represents not light to dark, but a relationship between black and brown

# look at core clusters
pca_clust1_data <- wings[wings$PC1 > -0.6 & wings$PC1 < -0.25 
                               & wings$PC2 > 0.01 & wings$PC2 < 0.4,]
table(pca_clust1_data$species)

# workaround to plot colors individually
wings$id <- seq(1,nrow(wings))
rgb_vect <- function(r_v, g_v, b_v,max){
  rgb_vect <- c()
  for(i in 1:length(r_v)){
    rgb_vect <- c(rgb_vect,rgb(r_v[i],g_v[i],b_v[i],maxColorValue = max))
  }
  return(rgb_vect)
}
wings$mean_hex <- rgb_vect(wings$mean_r, wings$mean_g, wings$mean_b, max = 1)

hex <- wings$mean_hex
names(hex) <- wings$id
ggplot(wings,aes(PC1,PC2)) + 
  geom_point(aes(colour=factor(id))) +
  scale_colour_manual(values = wings$mean_hex)  

#kmeans
library(factoextra)
bound <- cbind(wings$col_6_prop,wings$col_2_prop)
fviz_nbclust(bound,FUNcluster = kmeans)
km <- kmeans(bound, 3)
summary(km)
fviz_cluster(km, data = bound,
             geom = "point")
View(bound)

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
