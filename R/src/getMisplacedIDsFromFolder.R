getMisplacedIDsFromFolder <- function(dir="E:/dragonfly-patterner/data/patterns"){
  library(stringr)
  ids <- list.files(dir)
  ids <- unique(str_split_fixed(ids, "_", n =2)[,1])
  str <- ""
  for (i in ids){
    str <- paste0(str,i,sep='\",\"')
  }
  
  return(cat(str))
}

getIDsFromFolder <- function(dir="E:/dragonfly-patterner/data/patterns"){
  library(stringr)
  ids <- list.files(dir)
  ids <- unique(str_split_fixed(ids, "_", n =2)[,1])
  ids <- str_replace(ids,".jpg","")
  return(ids)
}

