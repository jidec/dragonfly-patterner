library(ape)
library(phytools)
library(sjPlot)
library(phyr)
library(rr2)

# more columns in wings_phy to mess around with in modeling
# 1. meanSe - mean seasonality
# 2. meanPrec - mean annual precipitation
# 3. various wing measurements
# 4. lat lons

# wings phy is the data
wings_phy <- read.csv("wings_phy.csv")
# phy is the phylogeny TRIMMED to match the data - in RDS to avoid save loading woes 
phy <- readRDS("phy.rds")

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

# black is color 6, brown is color 1, yellow is color 2
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
