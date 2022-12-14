
bound_records <- read.csv("E:/dragonfly-patterner/data/records.csv")
bound_records$imageID <- paste0("INAT-",gsub(".jpg","",bound_records$file_name))

aesh_bodies <- extractSimpleColorStats("E:/dragonfly-patterner/data/patterns/aesh_grouped5000",1)
aesh_bodies$mean_r <- as.numeric(aesh_bodies$mean_r)
aesh_bodies$mean_g <- as.numeric(aesh_bodies$mean_g)
aesh_bodies$mean_b <- as.numeric(aesh_bodies$mean_b)
aesh_bodies$col_1_prop <- as.numeric(aesh_bodies$col_1_prop)
aesh_bodies <- aesh_bodies[,-21:-24]

aesh_bodies$lightness <- (aesh_bodies$mean_r + aesh_bodies$mean_g + aesh_bodies$mean_b) / 3
hist(aesh_bodies$lightness)
aesh_bodies$imageID <- aesh_bodies$recordID

aesh_bodies <- merge(bound_records,aesh_bodies,all=TRUE,by="imageID")
library(dplyr)
aesh_bodies <- filter(aesh_bodies,!is.na(mean_r))

model <- lm(lightness ~ latitude,aesh_bodies)
summary(model)

ggplot(aesh_bodies_records, aes(x=latitude,y=lightness)) + geom_point() + geom_smooth(method='lm', formula= y~x)
labs(x="Training Set Size",y="Test Loss") 

library(ggplot2)
ggplot(aesh_bodies, aes(x=latitude,y=col_1_prop)) + geom_point() + geom_smooth(method='lm', formula= y~x)
labs(x="Training Set Size",y="Test Loss") 


path <- "E:/dragonfly-patterner/data/patterns/aesh_grouped5000/INAT-17788707-8_pattern.png"
# find the cluster given that its broken
img <- load.image(path)
arr <- as.array(img)
# reshape to pixels
dim(arr) <- c(dim(arr)[1] * dim(arr)[2],4)
# remove black background pixels 
arr <- arr[(arr != c(0,0,0,0))[,1],]

colors <- unique(arr)
colors <- colors[order(colors[,1],decreasing=FALSE),]
colors
colors * 255
blue_cluster <- c(0.5568627,0.6666667,0.7137255)
blue_cluster <- round(blue_cluster,5)
blue_cluster
green_cluster <- c(0.4862745,0.4784314,0.2588235)
green_cluster <- round(green_cluster,5)

aesh_bodies$col_4_r <- round(as.numeric(aesh_bodies$col_4_r),digits=5)
aesh_bodies$col_1_r <- round(as.numeric(aesh_bodies$col_1_r),digits=5)
aesh_bodies$col_2_r <- round(as.numeric(aesh_bodies$col_2_r),digits=5)
aesh_bodies$col_3_r <- round(as.numeric(aesh_bodies$col_3_r),digits=5)

sum(aesh_bodies$col_4_r == blue_cluster[1] | aesh_bodies$col_1_r == blue_cluster[1] | 
      aesh_bodies$col_2_r == blue_cluster[1] | aesh_bodies$col_3_r == blue_cluster[1],na.rm=TRUE)

aesh_bodies$has_blue <- aesh_bodies$col_4_r == blue_cluster[1] | aesh_bodies$col_1_r == blue_cluster[1] | 
  aesh_bodies$col_2_r == blue_cluster[1] | aesh_bodies$col_3_r == blue_cluster[1]

aesh_bodies$has_green <- aesh_bodies$col_4_r == green_cluster[1] | aesh_bodies$col_1_r == green_cluster[1] | 
  aesh_bodies$col_2_r == green_cluster[1] | aesh_bodies$col_3_r == green_cluster[1]

aesh_bodies$species <- aesh_bodies$taxon
rdpm <- removeDataPhyloMissing(aesh_bodies,odonate_tree)
aesh_rdpm <- rdpm[[1]]
#aesh_rdpm$has_blue <- as.factor(aesh_rdpm$has_blue)
aesh_rdpm_phy <- rdpm[[2]]
plot(aesh_rdpm_phy)

aesh_rdpm_species <- data.frame(aesh_rdpm %>%
  group_by(species) %>%
  summarise(has_blue = any(has_blue),has_green = any(has_green)))


aesh_rdpm_species$index <- match(aesh_rdpm_species$species,aesh_rdpm_phy$tip.label)
aesh_rdpm_species <- aesh_rdpm_species[order(aesh_rdpm_species$index), ]

aesh_rdpm_species$has_blue[is.na(aesh_rdpm_species$has_blue)] <- FALSE
aesh_rdpm_species$has_blue[aesh_rdpm_species$has_blue == TRUE] <- "blue"
aesh_rdpm_species$has_blue[aesh_rdpm_species$has_blue == FALSE] <- "none"
aesh_rdpm_species$has_blue_factor <- as.factor(aesh_rdpm_species$has_blue)

vect <- aesh_rdpm_species$has_blue_factor
names(vect) <- aesh_rdpm_species$species
vect
dotTree(aesh_rdpm_phy,vect,colors=setNames(c("blue","brown"),
                                       c("blue","none")),ftype="i",fsize=0.7)




aesh_rdpm_species$has_green[is.na(aesh_rdpm_species$has_green)] <- FALSE
aesh_rdpm_species$has_green[aesh_rdpm_species$has_green == TRUE] <- "green"
aesh_rdpm_species$has_green[aesh_rdpm_species$has_green == FALSE] <- "none"

vect <- as.factor(aesh_rdpm_species$has_green)
names(vect) <- aesh_rdpm_species$species
vect
dotTree(aesh_rdpm_phy,vect,colors=setNames(c("green","brown"),
                                           c("green","none")),ftype="i",fsize=0.7)


aesh_rdpm_species
row.names(aesh_rdpm_species) <- aesh_rdpm_species$species
#eel.tree<-read.tree("data/elopomorph.tre")
#eel.data<-read.csv("data/elopomorph.csv",row.names=1)
fmode <- as.factor(setNames(eel.data[,1],rownames(eel.data)))
fmode
fmode
aesh_rdpm_species$has_blue_factor
fmode
dotTree(eel.tree,fmode,colors=setNames(c("blue","red"),
                                       c("suction","bite")),ftype="i",fsize=0.7)