
# gomphids
gomph_bodies <- extractSimpleColorStats("E:/dragonfly-patterner/data/patterns/gomph_grouped_5000",1)
gomph_bodies$lightness <- rowMeans(gomph_bodies[,3:5])
gomph_bodies$mean_r <- as.numeric(gomph_bodies$mean_r)
gomph_bodies$mean_g <- as.numeric(gomph_bodies$mean_g)
gomph_bodies$mean_b <- as.numeric(gomph_bodies$mean_b)
gomph_bodies <- gomph_bodies[,-17:-20]

gomph_bodies$lightness <- (gomph_bodies$mean_r + gomph_bodies$mean_g + gomph_bodies$mean_b) / 3
hist(gomph_bodies$lightness)

bound_records$imageID <- paste0("INAT-",gsub(".jpg","",bound_records$file_name))

bound_records$imageID
gomph_bodies$imageID <- gomph_bodies$recordID
# merge onto records
gomph_bodies <- merge(bound_records,gomph_bodies,all=TRUE,by="imageID")
gomph_bodies <- filter(gomph_bodies_records,!is.na(mean_r))

gomph_bodies_records$col_1_prop <- as.numeric(gomph_bodies_records$col_1_prop)

gomph_bodies$inverse_lightness <- -1 * gomph_bodies$lightness

ggplot(gomph_bodies, aes(x=latitude,y=inverse_lightness)) + geom_point() + geom_smooth(method='lm', formula= y~x) +
labs(x="Latitude",y="Average HSL Darkness of Segment") 

ggplot(gomph_bodies, aes(x=latitude,y=col_1_prop)) + geom_point() + geom_smooth(method='lm', formula= y~x) +
  labs(x="Latitude",y="% Dark Color Cluster in Segment") 

