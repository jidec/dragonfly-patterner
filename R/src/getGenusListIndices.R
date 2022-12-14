
getGenusListIndices <- function(family,records_with_clades){
  records <- records_with_clades
  genus_list <- read.csv("E:/dragonfly-patterner/data/other/genus_list.csv")
  
  genera <- unique(records[records$family==family,]$genus)
  
  return(match(genera,genus_list$X0) - 1)
}