ids <- list.files("E:/dragonfly-patterner/data/patterns")
ids <- unique(str_split_fixed(ids, "_", n =2)[,1])
str <- ""
for (i in ids){
  str <- paste0(str,i,sep='\",\"')
}

cat(str)
