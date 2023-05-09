#m <- getPlotLM(wings_phy_trim,model_type="pglmm",response="black",phy=wings_phy,
#               formula="Sex * flight_type + (1 | species__)")
# m <- getPlotLM(wings,model_type="glmmTMB_beta_zi",response="black",
#                formula="wing_area * temp_indv + Sex + (1 | species)")
# m <- getPlotLM(wings,model_type="lm",response="black",
#                formula="wing_area * temp_indv ")

df <- wings

# get complete cases
df <- df[complete.cases(df$temp_indv),]
df <- df[complete.cases(df$wing_area),]

# model
m <- lm(black ~ wing_area + temp_indv, data=df)

# add resids to df
df$resids <- resid(m)

# minimal evidence of spat autocorr
c <- correlog(df$lon,df$lat,latlon = TRUE,z=df$resids,increment=500,resamp=20)
summary(c)
plot(c)
library(ggplot2)
library(ncf)
plot(c)
plot.correlog(c)
nrow(df)
ggplot(df) + geom_point(aes(x=lat,y=lon,color=resids))

library(ncf)
library(vegan)
library(geosphere)
mant <- ncf::mantel.test(M1=dist.temp,M2=dist.black,x=df$lat,y=df$lon,resamp=100)
?correlog

?ncf::mantel.test
geo = data.frame(df$lon, df$lat)
d.geo = distm(geo, fun = distHaversine)
dist.geo = as.dist(d.geo)

dist.black = dist(df$black, method = "euclidean")
dist.temp = dist(df$temp_indv, method = "euclidean")

temp_geo <- mantel.test()

temp_geo = mantel(dist.temp, dist.geo, method = "spearman", na.rm = TRUE)
temp_geo
?mantel
?mantel

write.csv(wings, "wings")
?write.csv

write.csv(wings_phy_trim, "wings_trim.csv")
saveRDS(wings_phy,"wings_phylo.rds")
write.csv(wings_phy, "wings_phylo.rds")
?write.csv

wings_phylo <- readRDS("wings_phylo.rds")
