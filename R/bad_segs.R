# counting number of bad segs 

sum(inferences$bad_signifier == 0,na.omit=TRUE)

test <- filter(inferences, bad_signifier == 1)
nrow(filter(inferences, bad_signifier == 1))
nrow(filter(inferences, bad_signifier == 0))

test <- filter(inferences, !is.na(bad_signifier))
test <- cbind(test$imageID,test$bad_signifier)
colnames(test) <- c("imageID","bad_signifier")
test <- data.frame(test)
test2 <- merge()