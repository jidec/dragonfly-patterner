
getSpPairwiseMetricDiffs <- function(color_data_sp,metric="lightness_mean"){
  wings_sp <- color_data_sp
  
  # get pairwise diffs in lightness 
  sp1 <- c()
  sp2 <- c()
  l_diffs <- c()
  c <- 0
  
  for(species in unique(wings_sp$clade)){
    
    sp1_l <- wings_sp[wings_sp$clade == species,][,metric]
    
    for(species2 in unique(wings_sp$clade)){
      sp2_l <- wings_sp[wings_sp$clade == species2,][,metric]
      
      diff <- abs(sp1_l - sp2_l)
      sp1 <- c(sp1,species)
      sp2 <- c(sp2,species2)
      l_diffs <- c(l_diffs,diff)
    }
    print(c)
    c <- c + 1
  }
  l_diffs <- data.frame(cbind(sp1,sp2,l_diffs))
  return(l_diffs)
}