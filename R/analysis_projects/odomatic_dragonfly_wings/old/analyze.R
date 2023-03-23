library(colorEvoHelpers)

## QUESTION: what is the most explanatory model predicting wing pigments and positioning?
# both pglmm and lmm? 
phy <- wings_phy

vifsAndStep(wings_phytrim)
wings_phytrim$flight_type <- as.factor(wings_phytrim$flight_type)
table(wings_phytrim$flight_type)

# pigment areas
phy <- wings_phy
wings_phy_trim$temp
wings_phy_trim$t


vifsAndStep(wings,response="black",formula="flight_type + wing_type + Sex + 
            temp_indv + temp_mean_sp + temp_sd_sp + temp_q5_sp + temp_q95_sp + 
            se_indv + se_mean_sp + 
            prec_indv + prec_mean_sp + Sex",mixed=FALSE)

wings$wing

vifsAndStep(wings_phy_trim,response="black",formula="flight_type + temp_indv + temp_mean_sp + temp_sd_sp + Sex",mixed=FALSE)
plotSaveLM(wings_phy_trim,"pglmm",response="black",formula="flight_type + temp_indv + temp_mean_sp + temp_sd_sp + Sex + (1 | species__)",scale = TRUE)
plotSaveLM(wings_phytrim,"pglmm",response="brown",formula="flight_type + Temp + Sex + (1 | species__)",scale = TRUE)
plotSaveLM(wings_phytrim,"pglmm",response="yellow",formula="flight_type + Temp + Sex + (1 | species__)",scale = TRUE)

# potentially physiologically meaningful pig combinations
# perchers have more brown/yellow, males have less brown/yellow
plotSaveLM(wings_phytrim,"pglmm",response="by_total",formula="flight_type + Temp + Sex + (1 | species__)",scale = TRUE)
# perchers probably have more brown/black, males have more brown/black
plotSaveLM(wings_phytrim,"pglmm",response="bb_total",formula="flight_type + Temp + Sex + (1 | species__)",scale = TRUE)
# perchers probably have more brown/black/yellow, males have slightly more brown/black/yellow
plotSaveLM(wings_phytrim,"pglmm",response="bby_total",formula="flight_type + Temp + Sex + (1 | species__)",scale = TRUE)

plotSaveLM(wings_phytrim,"pglmm",response="PC1_foreall",formula="flight_type + Temp + Sex + (1 | species__)",scale = TRUE)
plotSaveLM(wings_phytrim,"pglmm",response="PC2_foreall",formula="flight_type + Temp + Sex + (1 | species__)",scale = TRUE)
plotSaveLM(wings_phytrim,"pglmm",response="PC1_hindall",formula="flight_type + Temp + Sex + (1 | species__)",scale = TRUE)
plotSaveLM(wings_phytrim,"pglmm",response="PC2_hindall",formula="flight_type + Temp + Sex + (1 | species__)",scale = TRUE)


## QUESTION: do perchers have more pigments, and do perchers in colder environments have more pigments at the wing base? 
# perchers in colder environments should have more pigment at base of wings
# perchers in warmer environments should have pigments anywhere 
# if this is true, PCs which upweight base pigments should be correlated with flight style

plotSaveLM(filter(wings,Sex == "M"),model_type="lmer",
           response="bb_total",formula="flight_type + Temp + (1 | species)",
           plotmodel_terms=c(),scale = TRUE)

# in males, perchers have more black and individuals in colder environments have more black
plotSaveLM(filter(wings,Sex == "M"),model_type="lmer", response="black",formula="flight_type + temp_indv + (1 | species)", plotmodel_terms=c(),scale = TRUE)

# in females, indv in the cold still have more black, but no percher effect!!
plotSaveLM(filter(wings,Sex == "F"),model_type="lmer", response="black",formula="flight_type + temp_indv + (1 | species)", plotmodel_terms=c(),scale = TRUE)

# in males, perchers have more black and individuals in colder environments have more black
plotSaveLM(filter(wings,flight_type_rm_inter == "percher"),model_type="lmer", response="black",formula="temp_indv + (1 | species)", plotmodel_terms=c(),scale = TRUE)

plotSaveLM(filter(wings,flight_type_rm_inter == "flyer"),model_type="lmer", response="black",formula="temp_indv + (1 | species)", plotmodel_terms=c(),scale = TRUE)


plotSaveLM(wings,model_type="lmer", response="black",formula="temp_indv*flight_type_rm_inter + (1 | species)", plotmodel_terms=c(),scale = TRUE)

# perchers have more black pigments in colder env
# flyers do not 

# in females, indv in the cold still have more black, but no percher effect!!
plotSaveLM(filter(wings,Sex == "F"),model_type="lmer", response="black",formula="flight_type + temp_indv + (1 | species)", plotmodel_terms=c(),scale = TRUE)
# high vif between temp and interaction - gliders might just be in warmer generally 

plotSaveLM(wings,model_type="lmer", response="flight_type_rm_inter",formula="temp_indv + (1 | species)", plotmodel_terms=c(),scale = TRUE)
summary(glm(flight_type_rm_inter ~ temp_indv,data=wings,family="binomial"))

m <- glm(flight_type_rm_inter~1,data=wings,family="binomial")

plotLogReg(wings,response="flight_type_rm_inter",formula="temp_indv")
plot_model(m)
summary(glm(flight_type_rm_inter ~1,data=wings,family="binomial"))

summary(lm(temp_indv ~ flight_type_rm_inter,data=wings))

summary(aov(temp_indv ~ flight_type_rm_inter, data= wings))

# if using a binary, must use binomial 
# odds ratio -
# all glms have link function that fits complex shape
# finding the linear fit using the exponentiated 
# looking at the pat PC plots, the relevant PCs:
#   fore PC1, which has more pigment overall and especially at the base
#   fore PC2, which has more pigment at the base and less away from it
#   hind PC1, which has more pigment everywhere in a striped pattern

# pig base PC should be most effected by temperature 

# neg effect in hind PC1, which has more pigment everywhere in a striped pattern
plotSaveLM(filter(wings,Sex == "M",flight_type=="percher"),model_type="lmer", response="PC1_hindall",formula="temp_indv + (1 | species)", plotmodel_terms=c(),scale = TRUE)
# temperature increases fore PC1, which has more pigment overall and especially at the base
plotSaveLM(filter(wings,Sex == "M",flight_type=="percher"),model_type="lmer", response="PC1_foreall",formula="temp_indv + (1 | species)", plotmodel_terms=c(),scale = TRUE)
# temp decreases fore PC2, which has more pigment at the base and less away from it
plotSaveLM(filter(wings,Sex == "M",flight_type=="percher"),model_type="lmer", response="PC2_foreall",formula="temp_indv + (1 | species)", plotmodel_terms=c(),scale = TRUE)

# what does it look like when you plot these PCs percher vs flyer in males? 
# look at the saved plot created in prepWingData

## QUESTION - given that females are not trying to look territorial, do they have different signals than males? 
# either in what pigments are used or where they are positioned? 
# can we predict if an individual is male or female using pigment amounts or pigment PCs? 


summary(glm(Sex ~ brown + black + yellow + PC1_foreall + PC2_foreall + PC1_hindall + PC2_hindall, wings, family = binomial))

plotSaveRF(wings,response = "Sex",formula="brown + yellow + black + 
           PC1_hindall + PC2_hindall + PC1_foreall + PC2_foreall")

# what is more phylogenetically conserved, pigments or patterns? 
# what do the pigments and pattern pcs look like plotted on the phylo? 

# are more pigments closer to the base of the wing in general? 

# model selection
vifsAndStep(wings,response="black",formula="flight_type + wing_type + Temp + lat",mixed=FALSE)
vifsAndStep(wings,response="black",formula="flight_type + wing_type + Temp",mixed=FALSE)


# QUESTION 4. 
# do species overlapping in their ranges possess divergent wing patterns? 
# test if PCs are neg correlated with range overlap
# test if EMD is neg correlated with range overlap

source("src/getSpRangeOverlapsFromSDMs.R")
range_ol <- getSpRangeOverlapsFromSDMs(sdm_data)

source("src/getSpPairwiseMetricDiffs.R")

l_diffs <- getSpPairwiseMetricDiffs(wings_sp,"col_6_mean")
overlap_ldiffs <- merge(range_ol, l_diffs, by=c("sp1","sp2"))
overlap_ldiffs <- scaleDf(makeNumeric(overlap_ldiffs, colnames=c("overlap","l_diffs")),all_numerics = TRUE)

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
