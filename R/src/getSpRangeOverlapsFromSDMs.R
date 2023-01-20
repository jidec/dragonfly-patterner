
getSpRangeOverlapsFromSDMs <- function(sdm_data){
  # get range overlaps between species
  sp1 <- c()
  sp2 <- c()
  overlap <- c()
  c <- 0
  for(species in unique(sdm_data$binomial)){
    sp1_pts <- sdm_data[sdm_data$binomial == species,1:2]
    
    for(species2 in unique(sdm_data$binomial)){
      sp2_pts <- sdm_data[sdm_data$binomial == species2,1:2]
      #print("Finding intersection")
      n_common_pts <- nrow(generics::intersect(sp1_pts,sp2_pts))
      sp1 <- c(sp1,species)
      sp2 <- c(sp2,species2)
      overlap <- c(overlap,n_common_pts)
    }
    print(c)
    c <- c + 1
  }
  range_ol <- data.frame(cbind(sp1,sp2,overlap))
  range_ol$sp1 <- str_replace(range_ol$sp1,"_"," ")
  range_ol$sp2 <- str_replace(range_ol$sp2,"_"," ")
  return(range_ol)
}