
loss <- c(0.101374,0.0689607,0.097, 0.0541)
size <- c(160,297,356,462)
#f1 <- c(0.62556,0.641417,0.64043,)
loss <- c(0.101374,0.0689607, 0.0541)
size <- c(160,297,462)

seg_results <- data.frame(cbind(loss,size))
seg_results

library(ggplot2)
ggplot(seg_results, aes(x=size,y=loss)) + 
  geom_line() + labs(x="Training Set Size",y="Test Loss")
