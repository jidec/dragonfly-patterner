spTreeToGenusTree <- function(tree){
  tips <- c()
  for(genus in unique(str_split_fixed(tree$tip.label, " ", 2)[,1])) {
    tip <- grep(genus,tree$tip.label)[1]
    tips <- c(tips,tip)
  }
  tree <- keep.tip(tree,tips)
  return(tree)
}