source("src/extractSimpleColorStats.R")
dir <- "../data/segments"
img_locs <- paste0(dir,"/", setdiff(list.files(dir), list.dirs(recursive = FALSE, full.names = FALSE)))
img_locs <- img_locs[-1]
color_stats <- extractSimpleColorStats(img_locs)
mean_colors <- extractMeanColor(img_locs)


# gomphids
gomph_bodies <- extractSimpleColorStats("E:/dragonfly-patterner/data/patterns/gomph_grouped_5000",1)
gomph_bodies$lightness <- rowMeans(gomph_bodies[,3:5])
gomph_bodies$mean_r <- as.numeric(gomph_bodies$mean_r)
gomph_bodies$mean_g <- as.numeric(gomph_bodies$mean_g)
gomph_bodies$mean_b <- as.numeric(gomph_bodies$mean_b)

gomph_bodies$lightness <- (gomph_bodies$mean_r + gomph_bodies$mean_g + gomph_bodies$mean_b) / 3
hist(gomph_bodies$lightness)

bound_records$imageID <- paste0("INAT-",gsub(".jpg","",bound_records$file_name))

bound_records$imageID
gomph_bodies$imageID <- gomph_bodies$recordID
# merge onto records
gomph_bodies_records <- merge(bound_records,gomph_bodies,all=TRUE,by="imageID")
gomph_bodies_records <- filter(gomph_bodies_records,!is.na(mean_r))

model <- lm(lightness ~ latitude,gomph_bodies_records)
summary(model)

gomph_bodies_records$col_1_prop <- as.numeric(gomph_bodies_records$col_1_prop)

ggplot(gomph_bodies_records, aes(x=latitude,y=lightness)) + geom_point() + geom_smooth(method='lm', formula= y~x)
  labs(x="Training Set Size",y="Test Loss") 
  
ggplot(gomph_bodies_records, aes(x=latitude,y=col_1_prop)) + geom_point() + geom_smooth(method='lm', formula= y~x)
labs(x="Training Set Size",y="Test Loss") 

gomph_bodies <- gomph_bodies[,-17:-20]


#aesh_bodies$col_4_r == blue_cluster[1] & aesh_bodies$col_4_g == blue_cluster[2] & aesh_bodies$col_4_b == blue_cluster[3] 
