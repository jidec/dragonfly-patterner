
# do the following:
# 1. extract color stats from "discretized" image folder
# 2. merge in Odomatic records, percher-flyer data, measurements, fore-hind,
#   indv level county lat lon and bioclim data,
# 3. organize cols
# 4. make and merge pattern pcs, including plotting PCs
# 5. summarize to species

# returns a list of dataframes in list indices:
# 1. individual-level data
# 2. individual-level data trimmed to phy
# 3. phy trimmed to data
# 4. species-level data
# 5. species-level data trimmed to phy
# 6. species-sex level data
# 7. species-sex level data trimmed to phy
# 8. species-sex-forehind level data
# 9. species-sex level data trimmed to phy

# wing_img_loc = "D:/wing-color/data/patterns/grouped"
# patpca_img_loc = "D:/wing-color/data/patterns/grouped/size_normalized"
# tc1_1 = 0.4549
# tc1_2 = 0.7372
# tc1_3 = 0.1843
# patpca_flight_style=TRUE

# MUST start by manually loading in SDMs
# TODO add automatic filtering for sp with at least some pigment
prepWingData <- function(wing_img_loc="D:/wing-color/data/patterns/grouped2",
                         patpca_img_loc="D:/wing-color/data/patterns/grouped2/eis",
                         tc1_1 = 0.4549, tc1_2 = 0.7372, tc1_3 = 0.1843,
                         patpca_flight_style=TRUE){
  library(dplyr)
  library(stringr)
  library(sp)
  library(readr)
  library(colorEvoHelpers)

  Nearctic_ModelResults_5KM <- read_csv("D:/wing-color/data/Nearctic_ModelResults_5KM.csv")

  start_time <- Sys.time()

  # extract color stats from discretized wings
  source("src/extractSimpleColorStats.R")
  wing_colors <- extractSimpleColorStats(wing_img_loc)

  # load records and merge
  wings <- read.csv("D:/wing-color/data/records.csv")
  wings$county <- str_to_title(paste(wings$County,wings$State))
  wings$recordID <- wings$uniq_id
  wings$species <- wings$Species.name
  wings$Sex <- as.factor(wings$Sex)
  wings <- merge(wing_colors,wings,all=FALSE)

  # clearer color cols
  # this must get changed when clustering changes
  wings$brown <- wings$col_1_prop
  #wings$yellow <- wings$col_2_prop
  wings$black <- wings$col_3_prop

  # merge measurements
  measures <- read.csv("D:/wing-color/data/measurements.csv")
  measures$recordID <- measures$Project.Unique.ID
  wings <- merge(wings,measures,all.x=TRUE)

  # merge in percher flyer
  pf <- read.csv("data/percher_flyer.csv")
  colnames(pf) <- c("Taxon","John_Percher","John_Flyer","John_Intermediate","John_Reference",
                    "Jess_Percher","Jess_Flyer","Jess_Intermediate","Jess_Reference")
  pf <- pf[-1,]
  consensus <- c()
  for(i in 1:nrow(pf)){
    row <- pf[i,]
    if(row$John_Percher == "x" & row$Jess_Percher == "X"){
      consensus <- c(consensus,"percher")
    }
    else if(row$John_Flyer == "x" & row$Jess_Flyer == "X"){
      consensus <- c(consensus,"flyer")
    }
    else if(row$John_Intermediate == "x" & row$Jess_Intermediate == "X"){
      consensus <- c(consensus,"intermediate")
    }
    else{
      consensus <- c(consensus,NA)
    }
  }
  library(stringr)
  pf$species <- paste(str_split_fixed(pf$Taxon," ", 3)[,1],str_split_fixed(pf$Taxon," ", 3)[,2])
  pf$flight_type <- consensus
  wings <- left_join(wings,pf,by="species")

  # add wing_type col
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

  # merge county lat
  # add county lats where USA counties exist
  counties <- housingData::geoCounty
  counties$county <- str_to_title(paste(counties$rMapCounty,counties$rMapState))
  wings <- merge(wings,counties,all.x=TRUE)

  # get bioclim data and merge into wings
  r <- raster::getData("worldclim",var="bio",res=10)
  r <- r[[c(1,4,12)]]
  names(r) <- c("temp_indv","se_indv","prec_indv")

  values <- raster::extract(r, cbind(wings$lon,wings$lat))
  wings <- cbind(wings, values)

  # add some final cols
  wings$bb_total <- wings$brown + wings$black
  #wings$bby_total <- wings$brown + wings$black + wings$yellow
  #wings$by_total <- wings$brown + wings$yellow
  wings$fore_length <- wings$Length..inner..FW..mm.
  wings$tip_angle <-  wings$Tip.angle.FW....
  wings$wing_type <- as.factor(wings$wing_type)
  wings$latitude <- wings$lat
  wings$longitude <- wings$lon
  wings$flight_type <- as.factor(wings$flight_type)
  wings$flight_type_rm_inter <- wings$flight_type
  wings$flight_type_rm_inter[wings$flight_type_rm_inter == "intermediate"] <- NA
  wings$flight_type <- as.factor(wings$flight_type_rm_inter)

  # process and merge SDM data
  # get every 100th cell (making each cell represent about ~100 km)
  sdm_data <- Nearctic_ModelResults_5KM[seq(1, nrow(Nearctic_ModelResults_5KM), 100), ] #10
  sdm_data <- cbind(sdm_data,extract(r, SpatialPoints(cbind(sdm_data$x,sdm_data$y))))
  # get temp, seasonality, and prec means sds and quantiles
  sdm_data_sp <- sdm_data %>%
    group_by(binomial) %>%
    dplyr::summarise(temp_mean_sp = mean(temp_indv,na.rm=TRUE), temp_sd_sp = sd(temp_indv,na.rm=TRUE),
                     se_mean_sp = mean(se_indv,na.rm=TRUE), se_sd_sp = sd(se_indv,na.rm=TRUE),
                     prec_mean_sp = mean(prec_indv,na.rm=TRUE), prec_sd_sp = sd(prec_indv, na.rm=TRUE),
                     temp_q95_sp = quantile(temp_indv, 0.95,na.rm=TRUE), temp_q5_sp = quantile(temp_indv, 0.05,na.rm=TRUE),n=n())

  # merge sdm data to wing data
  sdm_data_sp$clade <- str_replace(sdm_data_sp$binomial,"_"," ")
  wings$clade <- wings$species
  wings <- merge(wings,sdm_data_sp,by="clade")

  # make pattern pcs
  source("src/makePatternPCA.R")
  # fore_pca_black <- makePatternPCA(wing_data=wings,pattern_dir=patpca_img_loc,
  #                                  wing_type="fore", filter_ids = NA,
  #                                  target_channel1_1=-1,target_channel1_2=-1,target_channel1_3=tc1_3,
  #                                  yPC=2,title="Predicted black pigment in fore",color_by_flight_style = patpca_flight_style)
  # fore_pca_brown <- makePatternPCA(wing_data=wings,pattern_dir=patpca_img_loc,
  #                                  wing_type="fore", filter_ids = NA,
  #                                  target_channel1_1=-1,target_channel1_2=tc1_2,target_channel1_3=-1,
  #                                  yPC=2,title="Predicted brown pigment in fore",color_by_flight_style = patpca_flight_style)
  # fore_pca_yellow <- makePatternPCA(wing_data=wings,pattern_dir=patpca_img_loc,
  #                                   wing_type="fore", filter_ids = NA,
  #                                   target_channel1_1=tc1_1,target_channel1_2=-1,target_channel1_3=-1,
  #                                   yPC=2,title="Predicted yellow pigment in fore",color_by_flight_style = patpca_flight_style)
  #
  # hind_pca_black <- makePatternPCA(wing_data=wings,pattern_dir=patpca_img_loc,
  #                                  wing_type="hind", filter_ids = NA,
  #                                  target_channel1_1=-1,target_channel1_2=-1,target_channel1_3=tc1_3,
  #                                  yPC=2,title="Predicted black pigment in hind",color_by_flight_style = patpca_flight_style)
  # hind_pca_brown <- makePatternPCA(wing_data=wings,pattern_dir=patpca_img_loc,
  #                                  wing_type="hind", filter_ids = NA,
  #                                  target_channel1_1=-1,target_channel1_2=tc1_2,target_channel1_3=-1,
  #                                  yPC=2,title="Predicted brown pigment in hind",color_by_flight_style = patpca_flight_style)
  # hind_pca_yellow <- makePatternPCA(wing_data=wings,pattern_dir=patpca_img_loc,
  #                                   wing_type="hind", filter_ids = NA,
  #                                   target_channel1_1=tc1_1,target_channel1_2=-1,target_channel1_3=-1,
  #                                   yPC=2,title="Predicted yellow pigment in hind",color_by_flight_style = patpca_flight_style)

  #hind_pca_all <- makePatternPCA(wing_data=wings,pattern_dir=patpca_img_loc,
  #                               wing_type="hind", filter_ids = NA,
  #                               target_channel1_1=tc1_1,target_channel1_2=tc1_2,target_channel1_3=tc1_3,
  #                               yPC=2,title="Predicted all pigment in hind",color_by_flight_style = patpca_flight_style)
  #fore_pca_all <- makePatternPCA(wing_data=wings,pattern_dir=patpca_img_loc,
  #                               wing_type="fore", filter_ids = NA,
  #                               target_channel1_1=tc1_1,target_channel1_2=tc1_2,target_channel1_3=tc1_3,
  #                               yPC=2,title="Predicted all pigment in fore",color_by_flight_style = patpca_flight_style)


  #hind_pca_all_cutoff <- makePatternPCA(wing_data=wings,pattern_dir=patpca_img_loc,
  #                               wing_type="hind", filter_ids = NA, color_percent_cutoff = 0.03,
  #                               target_channel1_1=tc1_1,target_channel1_2=tc1_2,target_channel1_3=tc1_3,
  #                               yPC=2,title="Predicted all pigment in hind",color_by_flight_style = patpca_flight_style)
  #fore_pca_all_cutoff <- makePatternPCA(wing_data=wings,pattern_dir=patpca_img_loc,
  #                               wing_type="fore", filter_ids = NA, color_percent_cutoff = 0.03,
  #                               target_channel1_1=tc1_1,target_channel1_2=tc1_2,target_channel1_3=tc1_3,
  #                               yPC=2,title="Predicted all pigment in fore",color_by_flight_style = patpca_flight_style)

  #source("src/mergePatPCToData.R")

  # wings <- mergePatPCToData(wings,fore_pca_all,"foreall")
  # wings <- mergePatPCToData(wings,hind_pca_black,"hindblack")
  # wings <- mergePatPCToData(wings,fore_pca_brown,"forebrown")
  # wings <- mergePatPCToData(wings,hind_pca_brown,"hindbrown")
  # wings <- mergePatPCToData(wings,fore_pca_yellow,"foreyellow")
  # wings <- mergePatPCToData(wings,hind_pca_yellow,"hindyellow")
  #wings <- mergePatPCToData(wings,fore_pca_all,"foreall")
  #wings <- mergePatPCToData(wings,hind_pca_all,"hindall")

  #wings <- mergePatPCToData(wings,fore_pca_all_cutoff,"foreallcut")
  #wings <- mergePatPCToData(wings,hind_pca_all_cutoff,"hindallcut")

  # summarize to sp
  wings_sp <- wings %>%
    group_by(species) %>%
    dplyr::summarise(black = mean(black),brown = mean(brown),#yellow = mean(yellow),
                     #all_fore_pc1=mean(PC1_foreall,na.rm=TRUE),all_fore_pc2=mean(PC2_foreall,na.rm=TRUE),
                     #all_hind_pc1=mean(PC1_hindall,na.rm=TRUE),all_hind_pc2=mean(PC2_hindall,na.rm=TRUE),
                     flight_type=Mode(flight_type),temp_mean_indv=mean(temp_indv,na.rm=TRUE),
                     temp_mean_sp = mean(temp_mean_sp), temp_sd_sp = mean(temp_sd_sp), # SDM daata
                     se_mean_sp = mean(se_mean_sp), se_sd_sp = mean(se_sd_sp),
                     prec_mean_sp = mean(prec_mean_sp), prec_sd_sp = mean(prec_sd_sp),
                     temp_q95_sp = mean(temp_q95_sp), temp_q95_sp = mean(temp_q95_sp),n=n())

  # summarize to sp sex
  wings_sp_sex <- wings %>%
    group_by(species,Sex) %>%
    dplyr::summarise(black = mean(black),brown = mean(brown),#yellow = mean(yellow),
                     #all_fore_pc1=mean(PC1_foreall,na.rm=TRUE),all_fore_pc2=mean(PC2_foreall,na.rm=TRUE),
                     #all_hind_pc1=mean(PC1_hindall,na.rm=TRUE),all_hind_pc2=mean(PC2_hindall,na.rm=TRUE),
                     flight_type=Mode(flight_type),temp_mean_indv=mean(temp_indv,na.rm=TRUE),
                     temp_mean_sp = mean(temp_mean_sp), temp_sd_sp = mean(temp_sd_sp), # SDM daata
                     se_mean_sp = mean(se_mean_sp), se_sd_sp = mean(se_sd_sp),
                     prec_mean_sp = mean(prec_mean_sp), prec_sd_sp = mean(prec_sd_sp),
                     temp_q95_sp = mean(temp_q95_sp), temp_q95_sp = mean(temp_q95_sp),n=n())

  # summarize to sp sex wing type
  wings_sp_sex_type <- wings %>%
    group_by(species,Sex,wing_type) %>%
    dplyr::summarise(black = mean(black),brown = mean(brown),#yellow = mean(yellow),
                     #all_fore_pc1=mean(PC1_foreall,na.rm=TRUE),all_fore_pc2=mean(PC2_foreall,na.rm=TRUE),
                     #all_hind_pc1=mean(PC1_hindall,na.rm=TRUE),all_hind_pc2=mean(PC2_hindall,na.rm=TRUE),
                     flight_type=Mode(flight_type),temp_mean_indv=mean(temp_indv,na.rm=TRUE),
                     temp_mean_sp = mean(temp_mean_sp), temp_sd_sp = mean(temp_sd_sp), # SDM daata
                     se_mean_sp = mean(se_mean_sp), se_sd_sp = mean(se_sd_sp),
                     prec_mean_sp = mean(prec_mean_sp), prec_sd_sp = mean(prec_sd_sp),
                     temp_q95_sp = mean(temp_q95_sp), temp_q95_sp = mean(temp_q95_sp),n=n())

  # summarize to sp trimmed to phy
  # load odonate tree
  wings_sp$clade <- wings_sp$species
  trimmed <- trimDfToTree(wings_sp,"E:/dragonfly-patterner/R/data/odonata.tre",replace_underscores = TRUE)
  wings_sp_phytrim <- trimmed[[1]]
  wings_sp_phy <- trimmed[[2]]

  # phy trim indv data
  wings$clade <- wings$species
  trimmed <- trimDfToTree(wings,"E:/dragonfly-patterner/R/data/odonata.tre",replace_underscores = TRUE)
  wings_phytrim <- trimmed[[1]]
  wings_phy <- trimmed[[2]]

  end_time <- Sys.time()
  print(end_time - start_time)

  wings$flight_type_rm_inter <- as.factor(as.character(wings$flight_type_rm_inter))
  wings_phytrim$flight_type_rm_inter <- as.factor(as.character(wings_phytrim$flight_type_rm_inter))


  # returns a list of dataframes in list indices:
  # 1. individual-level data
  # 2. individual-level data trimmed to phy
  # 3. phy trimmed to data
  # 4. species-level data
  # 5. species-level data trimmed to phy

  # 6. species-sex level data
  # 7. species-sex-forehind level data

  out <- list(wings,
              wings_phytrim,
              wings_phy,
              wings_sp,
              wings_sp_phytrim,
              wings_sp_sex,
              wings_sp_sex_type)
  saveRDS(out,"wing_data_tuple.rds")
  return(out)
}
