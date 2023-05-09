
wings2$Sex <- recode_factor(wings2$Sex, M = "Male", F = "Female")
wings2$black01 <- ifelse(wings2$black  > 0, 1, wings2$black)

wings2black <- dplyr::filter(wings2,black > 0.02)
wings2yellow <- dplyr::filter(wings2,yellow > 0.02)
wings2brown <- dplyr::filter(wings2,brown > 0.02)
wings$sex <- wings$Sex
