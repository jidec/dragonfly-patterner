data_tuple <- readRDS("wing_data_tuple.rds")

# returns a list of dataframes in list indices:
# 1. individual-level data
# 2. individual-level data trimmed to phy
# 3. phy trimmed to data
# 4. species-level data
# 5. species-level data trimmed to phy

# 6. species-sex level data
# 7. species-sex-forehind level data

wings <- data_tuple[[1]]
wings_phy_trim <- data_tuple[[2]]
wings_phy <- data_tuple[[3]]
phy <- wings_phy

wings_sp <- data_tuple[[4]]
wings_sp_phy_trim <- data_tuple[[5]]

wings_sp_sex <- data_tuple[[6]]
wings_sp_sex_type <- data_tuple[[7]]

# final repeatable transforms
wings$black01 <- wings$black > 0
wings$prop_black <- wings$black / wings$bby_total

wings_black <- dplyr::filter(wings,black > 0.02)
wings_brown <- dplyr::filter(wings,brown > 0.02)
wings_bb <- dplyr::filter(wings,brown > 0.02 | black > 0.02)
wings_yellow <- dplyr::filter(wings,yellow > 0.02)

wings$wing_area <- wings$Area.FW..mm2.

# summarize to sp sex
wings_sp <- wings %>%
    group_by(species) %>%
    dplyr::summarise(black = mean(black),brown = mean(brown),yellow = mean(yellow),
                     all_fore_pc1=mean(PC1_foreall,na.rm=TRUE),all_fore_pc2=mean(PC2_foreall,na.rm=TRUE),
                     all_hind_pc1=mean(PC1_hindall,na.rm=TRUE),all_hind_pc2=mean(PC2_hindall,na.rm=TRUE),

                     cut_fore_pc1=mean(PC1_foreallcut,na.rm=TRUE),cut_fore_pc2=mean(PC2_foreallcut,na.rm=TRUE),
                     cut_hind_pc1=mean(PC1_hindallcut,na.rm=TRUE),cut_hind_pc2=mean(PC2_hindallcut,na.rm=TRUE),

                     black_hind_pc1=mean(PC1_hindblack,na.rm=T),black_fore_pc1=mean(PC1_foreblack,na.rm=T),
                     black_hind_pc2=mean(PC2_hindblack,na.rm=T),black_fore_pc2=mean(PC2_hindblack,na.rm=T),
                     flight_type=Mode(flight_type),temp_mean_indv=mean(temp_indv,na.rm=TRUE),
                     temp_mean_sp = mean(temp_mean_sp), temp_sd_sp = mean(temp_sd_sp), # SDM daata
                     se_mean_sp = mean(se_mean_sp), se_sd_sp = mean(se_sd_sp),
                     prec_mean_sp = mean(prec_mean_sp), prec_sd_sp = mean(prec_sd_sp),
                     temp_q95_sp = mean(temp_q95_sp), temp_q95_sp = mean(temp_q95_sp),n=n())

wings_sp_na0 <- wings_sp %>%
    mutate(all_fore_pc1 = coalesce(all_fore_pc1, 0),
           all_fore_pc2 = coalesce(all_fore_pc2, 0),
           all_hind_pc1 = coalesce(all_hind_pc1, 0),
           all_hind_pc2 = coalesce(all_hind_pc2, 0),
           black_hind_pc1 = coalesce(black_hind_pc1, 0),
           black_hind_pc2 = coalesce(black_hind_pc2, 0),
           black_fore_pc1 = coalesce(black_fore_pc1, 0),
           black_fore_pc2 = coalesce(black_fore_pc2, 0),
           cut_fore_pc1 = coalesce(cut_fore_pc1, 0),
           cut_fore_pc2 = coalesce(cut_fore_pc2, 0),
           cut_hind_pc1 = coalesce(cut_hind_pc1, 0),
           cut_hind_pc2 = coalesce(cut_hind_pc2, 0))
wings_sp_na0$clade <- wings_sp_na0$species

wings_lib <- dplyr::filter(wings, Family=="Libellulidae")
wings_lib_black <- dplyr::filter(wings_black, Family=="Libellulidae")

wings_rm_cal <- dplyr::filter(wings, Family != "Calopterygidae")
wings$Family
