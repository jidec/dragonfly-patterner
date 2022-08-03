
countContains <- function(contains, direct_dir){
  library(stringr)
  return(length(str_subset(list.files(direct_dir),contains)))
}

countUniqueIDs <- function(direct_dir){
  library(stringr)
  out <- str_split_fixed(list.files(direct_dir),"_",n=2)
}

