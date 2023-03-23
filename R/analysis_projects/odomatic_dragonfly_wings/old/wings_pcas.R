wing_colors[1,]
tc1_1 <- 0.4549
tc1_2 <- 0.7372
tc1_3 <- 0.1843

source("src/makePatternPCA.R")
fore_pca_black <- makePatternPCA("D:/wing-color/data/patterns/grouped2/eis",
                                 wing_type="fore", filter_ids = NA,
                                 target_channel1_1=-1,target_channel1_2=-1,target_channel1_3=tc1_3,
                                 yPC=2,title="Predicted black pigment in fore")
fore_pca_brown <- makePatternPCA("D:/wing-color/data/patterns/grouped2/eis",
                                 wing_type="fore", filter_ids = NA,
                                 target_channel1_1=-1,target_channel1_2=tc1_2,target_channel1_3=-1,
                                 yPC=2,title="Predicted brown pigment in fore")
fore_pca_yellow <- makePatternPCA("D:/wing-color/data/patterns/grouped2/eis",
                                 wing_type="fore", filter_ids = NA,
                                 target_channel1_1=tc1_1,target_channel1_2=-1,target_channel1_3=-1,
                                 yPC=2,title="Predicted yellow pigment in fore")

hind_pca_black <- makePatternPCA("D:/wing-color/data/patterns/grouped2/eis",
                                 wing_type="hind", filter_ids = NA,
                                 target_channel1_1=-1,target_channel1_2=-1,target_channel1_3=tc1_3,
                                 yPC=2,title="Predicted black pigment in hind")
hind_pca_brown <- makePatternPCA("D:/wing-color/data/patterns/grouped2/eis",
                                 wing_type="hind", filter_ids = NA,
                                 target_channel1_1=-1,target_channel1_2=tc1_2,target_channel1_3=-1,
                                 yPC=2,title="Predicted brown pigment in hind")
hind_pca_yellow <- makePatternPCA("D:/wing-color/data/patterns/grouped2/eis",
                                  wing_type="hind", filter_ids = NA,
                                  target_channel1_1=tc1_1,target_channel1_2=-1,target_channel1_3=-1,
                                  yPC=2,title="Predicted yellow pigment in hind")

hind_pca_all <- makePatternPCA("D:/wing-color/data/patterns/grouped2/eis",
                               wing_type="hind", filter_ids = NA,
                               target_channel1_1=tc1_1,target_channel1_2=tc1_2,target_channel1_3=tc1_3,
                               yPC=2,title="Predicted all pigment in hind")
fore_pca_all <- makePatternPCA(wing_data=wings,pattern_dir="D:/wing-color/data/patterns/grouped2/eis",
                               wing_type="fore", filter_ids = NA,
                               target_channel1_1=tc1_1,target_channel1_2=tc1_2,target_channel1_3=tc1_3,
                               yPC=2,title="Predicted all pigment in fore")

source("src/mergePatPCToData.R")
wings2 <- mergePatPCToData(wings,fore_pca_all,"foreall")
wings2 <- mergePatPCToData(wings2,hind_pca_black,"hindblack")
wings2 <- mergePatPCToData(wings2,fore_pca_brown,"forebrown")
wings2 <- mergePatPCToData(wings2,hind_pca_brown,"hindbrown")
wings2 <- mergePatPCToData(wings2,fore_pca_yellow,"foreyellow")
wings2 <- mergePatPCToData(wings2,hind_pca_yellow,"hindyellow")

wings2 <- mergePatPCToData(wings,fore_pca_all,"foreall")
wings2 <- mergePatPCToData(wings2,hind_pca_all,"hindall")


lm(PC1_foreall ~ Sex,data=wings2)
wings2$PC1_foreall

t.test(PC1_foreblack ~ Sex,data=wings2)
t.test(PC2_foreblack ~ Sex,data=wings2) #sig
t.test(PC1_hindblack ~ Sex,data=wings2)
t.test(PC2_hindblack ~ Sex,data=wings2)

t.test(PC1_forebrown ~ Sex,data=wings2)
t.test(PC2_forebrown ~ Sex,data=wings2)
t.test(PC1_hindbrown ~ Sex,data=wings2)
t.test(PC2_hindbrown ~ Sex,data=wings2) #sig

t.test(PC1_foreyellow ~ Sex,data=wings2) # sig
t.test(PC2_foreyellow ~ Sex,data=wings2)# sig
t.test(PC1_hindyellow ~ Sex,data=wings2) # sig
t.test(PC2_hindyellow ~ Sex,data=wings2) # big sig

# phylo cont map
wings2$species <- as.factor(wings2$species)
wings2$PC1_foreblack <- as.numeric(wings2$PC1_foreblack)
wings2 <- wings2 %>%
  filter(Sex == 'M') %>%
  filter(!is.na(PC1_foreblack)) %>%
  group_by(species) %>%
  summarise(PC1_foreblack = mean(PC1_foreblack,na.rm=TRUE))
colnames(wings2) <- c("clade","trait")
library(ape)
tree < - read.tree("data/odonata.tre")
tree$tip.label <- str_replace(tree$tip.label,"_"," ")
wings2$clade <- as.character(wings2$clade)
plotPhyloEffects(wings2,tree,tipnames = FALSE)

