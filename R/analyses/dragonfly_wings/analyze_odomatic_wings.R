library(ape)
library(stringr)
library(dplyr)
library(raster)
library(sp)
library(lme4)
library(sjPlot)
library(car)
library(lmerTest)
library(performance)
library(ggplot2)
library(lattice)
library(ggbiplot)
library(phyr)
library(prediction)
library(housingData)
library(rr2)

# extract color stats from discretized wings
source("src/extractSimpleColorStats.R")
wing_colors <- extractSimpleColorStats("D:/wing-color/data/patterns/grouped2")

# load records and merge 
wings <- read.csv("D:/wing-color/data/records.csv")
wings$county <- str_to_title(paste(wings$County,wings$State))
wings$recordID <- wings$uniq_id
wings$species <- wings$Species.name
wings$Sex <- as.factor(wings$Sex)
wings <- merge(wing_colors,wings,all=FALSE)

# merge measurements
measures <- read.csv("D:/wing-color/data/measurements.csv")
measures$recordID <- measures$Project.Unique.ID
wings <- merge(wings,measures,all.x=TRUE)

# find lone wings
lone_wing_ids <- names(table(wings$recordID)[table(wings$recordID) == 1])
lone_wings_indices <- match(lone_wing_ids,wings$recordID)
lone_wings <- wings[lone_wings_indices,]
lone_wings$wing_type <- NA
wings <- wings[-lone_wings_indices,]
# add wing type using rep
wings$wing_type <- rep(c("fore","hind"), times=nrow(wings)/2, each=1)
# add lone wings back
wings <- rbind(wings,lone_wings)

# add county lats where USA counties exist
counties <- housingData::geoCounty
counties$county <- str_to_title(paste(counties$rMapCounty,counties$rMapState))
wings <- merge(wings,counties,all.x=TRUE)
wings_sp <- wings %>%
  group_by(Species.name) %>%
  dplyr::summarise(col_6_mean = mean(col_6_prop),col_1_mean = mean(col_1_prop),col_2_mean = mean(col_2_prop),lightness_mean = mean(mean_lightness))

# load odonate tree
odonate_tree <- ape::read.tree("E:/dragonfly-patterner/R/data/odonata.tre")
odonate_tree$tip.label <-  str_replace(odonate_tree$tip.label,"_"," ")
data_phy <- removeDataPhyloMissing(wings,odonate_tree)
wings_phy <- data_phy[[1]]
phy <- data_phy[[2]]

wings_phy_sp <- wings_phy %>%
    group_by(species) %>%
    dplyr::summarise(col_6_mean = mean(col_6_prop),col_1_mean = mean(col_1_prop),col_2_mean = mean(col_2_prop),lightness_mean = mean(mean_lightness))

# sdms
# IMPORT SDMS MANUALLY THROUGH "IMPORT DATASET" BUTTON

# get bioclim data
r <- raster::getData("worldclim",var="bio",res=10)
r <- r[[c(1,4,12)]]
names(r) <- c("Temp","Se","Prec")

# get every 100th cell (making each cell represent about ~100 km)
sdm_data <- Nearctic_ModelResults_5KM[seq(1, nrow(Nearctic_ModelResults_5KM), 100), ] #10
sdm_data <- cbind(sdm_data,extract(r, SpatialPoints(cbind(sdm_data$x,sdm_data$y))))

sdm_data_sp <- sdm_data %>%
  group_by(binomial) %>%
  dplyr::summarise(meanTemp = mean(Temp,na.rm=TRUE), sdTemp = sd(Temp,na.rm=TRUE), 
                   meanSe = mean(Se,na.rm=TRUE), sdSe = sd(Se,na.rm=TRUE), 
                   meanPrec = mean(Prec,na.rm=TRUE), sdPrec = sd(Prec, na.rm=TRUE),
            q95Temp = quantile(Temp, 0.95,na.rm=TRUE), q5Temp = quantile(Temp, 0.05,na.rm=TRUE),n=n())

# merge sdm data to wing data
sdm_data_sp$clade <- str_replace(sdm_data_sp$binomial,"_"," ")
wings$clade <- wings$species
wings_phy$clade <- wings_phy$Species.name
wings_sp$clade <- wings_sp$Species.name
wings_phy_sp$clade <- wings_phy_sp$species

wings <- merge(wings,sdm_data_sp)
wings_phy <- merge(wings_phy,sdm_data_sp)
wings_sp <- merge(wings_sp,sdm_data_sp)
wings_phy_sp <- merge(wings_phy_sp,sdm_data_sp)

#wings_sdm_data_sp <- wings_sdm_data_sp[wings_sdm_data_sp$col_1_mean > 0.05,]

# col 1 is brown
# col 2 is yellow
# col 6 is black

#wings_sdm_data_sp$pigs_mean <- wings_sdm_data_sp$col_6_mean + wings_sdm_data_sp$col_1_mean #+ wings_sdm_data_sp$col_2_mean
#wings_sdm_data_sp$col_6_mean[wings_sdm_data_sp$col_6_mean < 0.05] <- 0

# plot hist of number of obs of each species
hist(table(wings_phy$clade))

# mean temp is strongly correlated with mean sesonality, but sd is not
cor.test(wings_phy$meanTemp,wings_phy$meanSe)
cor.test(wings_phy$meanTemp,wings_phy$sdTemp)
cor.test(wings_phy$sdTemp,wings_phy$meanSe)

# thermal breadth is partially correlated with range area
cor.test(wings_phy$sdTemp,wings_phy$n)

# add some final cols
wings_phy$bb_total <- wings_phy$col_6_prop + wings_phy$col_1_prop #+ wings_sdm_data_sp$col_2_mean
wings_phy$bby_total <- wings_phy$col_6_prop + wings_phy$col_1_prop + wings_phy$col_2_prop#+ wings_sdm_data_sp$col_2_mean
wings_phy$by_total <- wings_phy$col_1_prop + wings_phy$col_2_prop
wings_phy$fore_length <- wings_phy$Length..inner..FW..mm.
wings_phy$tip_angle <-  wings_phy$Tip.angle.FW....
wings_phy$wing_type <- as.factor(wings_phy$wing_type)

# scale 
wings_phy_sc <- transform(wings_phy,
                  meanTemp=scale(meanTemp),
                  sdTemp=scale(sdTemp),
                  fore_length=scale(fore_length),
                  meanPrec=scale(meanPrec),
                  meanSe=scale(meanSe),
                  col_6_prop = scale(col_6_prop),
                  col_2_prop = scale(col_2_prop),
                  col_1_prop = scale(col_1_prop),
                  n=scale(n),
                  bb_total = scale(bb_total),
                  by_total = scale(by_total))

model_black <- pglmm(col_6_prop ~ 0 + meanTemp + sdTemp + wing_type + fore_length + Sex + (1 | species__),
                       data = wings_phy_sc, cov_ranef = list(species=phy),bayes=T)
model_brown <- pglmm(col_1_prop ~ 0 + meanTemp + sdTemp + wing_type + fore_length + Sex + (1 | species__),
                     data = wings_phy_sc, cov_ranef = list(species=phy),bayes=T)
model_yellow <- pglmm(col_2_prop ~ 0 + meanTemp + sdTemp + wing_type + fore_length + Sex + (1 | species__),
                     data = wings_phy_sc, cov_ranef = list(species=phy),bayes=T)
model_bb <- pglmm(bb_total ~ 0 + meanTemp + sdTemp + wing_type + fore_length + Sex + (1 | species__),
                      data = wings_phy_sc, cov_ranef = list(species=phy),bayes=T)
model_by <- pglmm(by_total ~ 0 + meanTemp + sdTemp + wing_type + fore_length + Sex + (1 | species__),
                  data = wings_phy_sc, cov_ranef = list(species=phy),bayes=T)

# plot 
plot_bayes(model_black) + ggtitle("Proportion of black pigment")
plot_bayes(model_brown) + ggtitle("Proportion of brown pigment")
plot_bayes(model_yellow) + ggtitle("Proportion of yellow pigment")
plot_bayes(model_bb) + ggtitle("Proportion of black and brown pigment")
plot_bayes(model_b) + ggtitle("Proportion of yellow and brown pigment")

# calculate r2 
rr2::R2(model_black)
rr2::R2(model_brown)
rr2::R2(model_yellow)

# plot cont maps
wings_phy_sp2 <- wings_phy_sp
colnames(wings_phy_sp2)[1] <- "clade"
colnames(wings_phy_sp2)[3] <- "trait"
plotPhyloEffects(wings_phy_sp2,phy)
colnames(wings_phy_sp2)[2] <- "trait"
plotPhyloEffects(wings_phy_sp2,phy)

# get dfs of pairwise range overlaps and metric diffs
source("src/getSpRangeOverlapsFromSDMs.R")
range_ol <- getSpRangeOverlapsFromSDMs(sdm_data)

source("src/getSpPairwiseMetricDiffs.R")

l_diffs <- getSpPairwiseMetricDiffs(wings_sp,"col_6_mean")
overlap_ldiffs <- merge(range_ol, l_diffs, by=c("sp1","sp2"))
overlap_ldiffs$overlap <- as.numeric(overlap_ldiffs$overlap)
overlap_ldiffs$l_diffs <- as.numeric(overlap_ldiffs$l_diffs)

overlap_ldiffs <- transform(overlap_ldiffs,
                          overlap=scale(overlap),
                          l_diffs=scale(l_diffs))
# test if range overlap is correlated with differences in the amount of black pigment
# realistically I should use EMD or similar pattern distance metric 
cor.test(overlap_ldiffs$overlap,overlap_ldiffs$l_diffs)
ggplot(overlap_ldiffs, aes(x=overlap, y=l_diffs)) + geom_point()

# compute correlation between traits on phylo per http://blog.phytools.org/2017/08/pearson-correlation-with-phylogenetic.html
t <- cbind(wings_phy_sp$col_6_mean,wings_phy_sp$col_2_mean,wings_phy_sp$col_1_mean)
rownames(t) <- wings_phy_sp$clade
colnames(t) <- c("black","yellow","brown")
phy2 <- drop.tip(phy,phy$tip.label[!phy$tip.label %in% rownames(t)])
obj<-phyl.vcv(t,vcv(phy2),1)
## correlation between x & y
r.xy<-cov2cor(obj$R)["black","brown"]
## t-statistic & P-value
t.xy<-r.xy*sqrt((Ntip(phy2)-2)/(1-r.xy^2))
P.xy<-2*pt(abs(t.xy),df=Ntip(phy2)-2,lower.tail=F)
r.xy
P.xy
# strong 0.47 correlation miniscule p between brown and black

row <- wing_colors[1,]
plotRGBColor(c(row$col_1_r,row$col_1_g,row$col_1_b),max=1)
plotRGBColor(c(row$col_2_r,row$col_2_g,row$col_2_b),max=1)
plotRGBColor(c(row$col_6_r,row$col_6_g,row$col_6_b),max=1)

# plot heatmaps using patternize 
