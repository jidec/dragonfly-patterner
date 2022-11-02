dl_aesh <- filter(merged, family=="Aeshnidae")
dl_matched_aesh <- filter(merged, has_downloaded_image == TRUE, gbifID!=-1,family=="Aeshnidae")
dl_nomatch_aesh <- filter(merged, has_downloaded_image == TRUE, gbifID==-1,family=="Aeshnidae")
merged$gbifID

table(matches$species)
plot(matches$decimalLongitude,matches$decimalLatitude)
plot(merged$decimalLatitude,merged$decimalLongitude)
plot(dl_nomatch_aesh$decimalLongitude,dl_nomatch_aesh$decimalLatitude)

nrow(filter(inat_records,catalogNumber==137088394))
